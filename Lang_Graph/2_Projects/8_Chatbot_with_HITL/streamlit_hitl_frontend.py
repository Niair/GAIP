# chainlit_app.py
import chainlit as cl
import uuid
from typing import List, Dict, Any, Optional
import asyncio

# Import your backend functions (from uploaded files)
from langgraph_hitl_backend import (
    initialize,
    chat_stream,
    process_document,
    remove_document,
    get_rag_status,
    get_all_threads,
    get_conversation_history,
    get_pending_tools,
    approve_tool_execution,
    reject_tool_execution,
)

# -----------------------
# Utilities
# -----------------------
def gen_thread_id() -> str:
    return str(uuid.uuid4())

async def ensure_session_keys():
    """Ensure user_session contains necessary keys."""
    if cl.user_session.get("thread_id") is None:
        cl.user_session.set("thread_id", gen_thread_id())
    if cl.user_session.get("conversation_history") is None:
        cl.user_session.set("conversation_history", [])
    if cl.user_session.get("awaiting_approval") is None:
        cl.user_session.set("awaiting_approval", False)
    if cl.user_session.get("pending_tools") is None:
        cl.user_session.set("pending_tools", None)
    if cl.user_session.get("thread_titles") is None:
        cl.user_session.set("thread_titles", {})

# -----------------------
# On Chat Start
# -----------------------
@cl.on_chat_start
async def on_chat_start():
    # Initialize backend
    await initialize()

    # Prepare session state
    await ensure_session_keys()

    # Create a fresh thread id for this session if not present
    if not cl.user_session.get("thread_id"):
        cl.user_session.set("thread_id", gen_thread_id())

    # Welcome + Commands
    await cl.Message(
        content=(
            "👋 **Welcome to AI Chatbot with MCP & HITL**\n\n"
            "I can help you with:\n"
            "• 📄 Document Q&A — upload a PDF to enable RAG\n"
            "• 🔍 Web Search\n"
            "• 💰 Finance tools (income/expenses)\n"
            "• 🧮 Math (MCP tools)\n"
            "• 🤝 Human-in-the-Loop approvals — approve tools before execution\n\n"
            "Use the command buttons or type `/help` to see available text commands."
        )
    ).send()

    # Add commands (they appear near the chat input in Chainlit UI)
    commands = [
        {"id": "UploadPDF", "icon": "upload-cloud", "description": "Upload PDF"},
        {"id": "RemoveDoc", "icon": "trash-2", "description": "Remove current document"},
        {"id": "NewChat", "icon": "plus-circle", "description": "Start a new chat"},
        {"id": "ListChats", "icon": "list", "description": "List saved chats"},
        {"id": "RenameChat", "icon": "edit-2", "description": "Rename current chat"},
        {"id": "Status", "icon": "info", "description": "Show RAG/document status"},
    ]
    # set_commands is optional depending on Chainlit version - safe attempt
    try:
        await cl.context.emitter.set_commands(commands)
    except Exception:
        # If API not available, it's okay; commands still work via typed slash commands
        pass

# -----------------------
# Helper: show tool approval flow
# -----------------------
async def show_tool_approval_ui(thread_id: str, tools: List[Dict[str, Any]]):
    """
    Present each pending tool to the user and ask approve/reject.
    This mirrors your previous Chainlit HITL flow but is centralized here.
    """
    for tool_info in tools:
        tool_name = tool_info.get('name', 'tool')
        tool_args = tool_info.get('args', {})

        # Friendly display names
        if tool_name == "rag_tool":
            icon, display_name = "📚", "Document Search (RAG)"
        elif tool_name == "duckduckgo_search":
            icon, display_name = "🔍", "Web Search"
        elif tool_name in ["add", "subtract", "multiply", "divide", "power", "modulus"]:
            icon, display_name = "🧮", f"Math: {tool_name.title()}"
        elif "expense" in tool_name or "income" in tool_name:
            icon, display_name = "💰", f"Finance: {tool_name.replace('_',' ').title()}"
        else:
            icon, display_name = "🔧", tool_name.replace('_', ' ').title()

        # Format args
        if isinstance(tool_args, dict):
            args_str = "\n".join(f"• **{k}**: `{str(v)[:200]}`" for k, v in tool_args.items())
        else:
            args_str = f"`{str(tool_args)}`"

        content = (
            f"⏸️ **Tool Execution Paused**\n\n"
            f"{icon} **{display_name}**\n\n"
            f"**Arguments:**\n{args_str}\n\n"
            "Do you want to execute this tool?"
        )

        # Show the paused-tool message
        await cl.Message(content=content).send()

        # Ask user to approve / reject
        res = await cl.AskActionMessage(
            content="Choose an action:",
            actions=[
                cl.Action(name="approve", value="approve", description="✅ Approve & Execute"),
                cl.Action(name="reject", value="reject", description="❌ Reject Tool"),
            ],
        ).send()

        if res and res.get("value") == "approve":
            # Execute approval flow similar to your backend
            await cl.Message(content="⚙️ Executing tool...").send()
            try:
                # stream approval result
                full = ""
                async for chunk in approve_tool_execution(thread_id):
                    if hasattr(chunk, 'content') and chunk.content:
                        # send streaming tokens as messages (small chunks)
                        await cl.Message(content=chunk.content).send()
                        full += chunk.content
                # final message
                await cl.Message(content="✅ Tool executed successfully.").send()
            except Exception as e:
                await cl.Message(content=f"❌ Error executing tool: {e}").send()
        else:
            # Reject
            try:
                res = await reject_tool_execution(thread_id, "User rejected the tool")
                if res.get('success'):
                    await cl.Message(content=f"⛔ Tool rejected: {res.get('message','') }").send()
                else:
                    await cl.Message(content=f"❌ Error rejecting tool: {res.get('error','unknown')}").send()
            except Exception as e:
                await cl.Message(content=f"❌ Error rejecting tool: {e}").send()

    # Reset approval session flags
    cl.user_session.set("awaiting_approval", False)
    cl.user_session.set("pending_tools", None)

# -----------------------
# On Message Handler (main)
# -----------------------
@cl.on_message
async def on_message(message: cl.Message):
    # Ensure session keys exist
    await ensure_session_keys()

    thread_id: str = cl.user_session.get("thread_id")
    user_text = (message.content or "").strip()

    # If we are in "awaiting approval" state, block new messages
    if cl.user_session.get("awaiting_approval"):
        await cl.Message(
            content="⏸️ Waiting for tool approval. Please approve or reject the pending tool before sending new messages."
        ).send()
        return

    # -----------------------
    # Handle built-in commands triggered by UI buttons
    # -----------------------
    if message.command:
        cmd = message.command
        # Upload via AskFileMessage
        if cmd == "UploadPDF":
            files = await cl.AskFileMessage(
                content="📤 Please upload a PDF file to process (will enable Document Q&A):",
                accept=["application/pdf"]
            ).send()
            if not files:
                await cl.Message(content="No file uploaded.").send()
                return
            pdf_path = files[0].path
            try:
                result = process_document(pdf_path)
                if result.get('success'):
                    info = result['info']
                    await cl.Message(
                        content=(
                            f"✅ **Document Loaded:** {info['filename']}\n\n"
                            f"📄 Pages: {info['pages']}    🔗 Chunks: {info['chunks']}"
                        )
                    ).send()
                else:
                    await cl.Message(content=f"❌ Error processing PDF: {result.get('error','Unknown')}").send()
            except Exception as e:
                await cl.Message(content=f"❌ Error processing PDF: {e}").send()
            return

        if cmd == "RemoveDoc":
            try:
                res = remove_document()
                if res.get('success'):
                    await cl.Message(content="🗑️ Document removed. RAG disabled.").send()
                else:
                    await cl.Message(content=f"❌ Error removing document: {res.get('error','Unknown')}").send()
            except Exception as e:
                await cl.Message(content=f"❌ Error: {e}").send()
            return

        if cmd == "NewChat":
            new_id = gen_thread_id()
            cl.user_session.set("thread_id", new_id)
            cl.user_session.set("conversation_history", [])
            await cl.Message(content="🔄 New chat session started.").send()
            return

        if cmd == "ListChats":
            # Query backend for threads
            try:
                threads = get_all_threads() or []
            except Exception as e:
                threads = []
                await cl.Message(content=f"❌ Error listing chats: {e}").send()

            if not threads:
                await cl.Message(content="ℹ️ No saved chats found.").send()
                return

            # Build a nice list with small actions (switch/preview)
            md_lines = []
            titles = cl.user_session.get("thread_titles") or {}
            for tid in threads[::-1]:
                title = titles.get(tid, "New Chat")
                md_lines.append(f"- **{title}** — `{tid}`")
            list_md = "\n".join(md_lines)
            await cl.Message(content=f"🗂️ **Saved Chats**\n\n{list_md}").send()
            # Offer a quick action to switch by asking for a thread id
            choice = await cl.AskTextMessage(
                content="If you want to switch to a chat, paste its *thread id* (or leave blank):",
                placeholder="paste thread id here"
            ).send()
            if choice and choice.strip():
                chosen = choice.strip()
                # Validate chosen thread exists
                if chosen in threads:
                    cl.user_session.set("thread_id", chosen)
                    cl.user_session.set("conversation_history", [])
                    await cl.Message(content=f"✅ Switched to thread `{chosen}`").send()
                else:
                    await cl.Message(content="❌ Thread id not found.").send()
            return

        if cmd == "RenameChat":
            current = cl.user_session.get("thread_id")
            current_titles = cl.user_session.get("thread_titles") or {}
            current_name = current_titles.get(current, "New Chat")
            name = await cl.AskTextMessage(
                content=f"Rename current chat (id `{current}`):",
                placeholder=current_name
            ).send()
            if name and name.strip():
                current_titles[current] = name.strip()
                cl.user_session.set("thread_titles", current_titles)
                await cl.Message(content=f"✏️ Chat renamed to **{name.strip()}**").send()
            return

        if cmd == "Status":
            rag = get_rag_status()
            if rag.get('has_document'):
                info = rag['document_info']
                await cl.Message(content=(
                    f"📚 **Document:** {info['filename']}\n"
                    f"📄 Pages: {info['pages']}\n"
                    f"🔗 Chunks: {info['chunks']}"
                )).send()
            else:
                await cl.Message(content="ℹ️ No document loaded.").send()
            return

    # -----------------------
    # Handle typed slash commands or text commands
    # -----------------------
    low = user_text.lower()
    if low.startswith("/upload"):
        # same as UploadPDF
        files = await cl.AskFileMessage(
            content="📤 Please upload a PDF file to process (will enable Document Q&A):",
            accept=["application/pdf"]
        ).send()
        if files:
            pdf_path = files[0].path
            try:
                result = process_document(pdf_path)
                if result.get('success'):
                    info = result['info']
                    await cl.Message(
                        content=(
                            f"✅ **Document Loaded:** {info['filename']}\n\n"
                            f"📄 Pages: {info['pages']}    🔗 Chunks: {info['chunks']}"
                        )
                    ).send()
                else:
                    await cl.Message(content=f"❌ Error processing PDF: {result.get('error','Unknown')}").send()
            except Exception as e:
                await cl.Message(content=f"❌ Error processing PDF: {e}").send()
        else:
            await cl.Message(content="No file uploaded.").send()
        return

    if low.startswith("/remove"):
        try:
            res = remove_document()
            if res.get('success'):
                await cl.Message(content="🗑️ Document removed.").send()
            else:
                await cl.Message(content=f"❌ Error: {res.get('error','Unknown')}").send()
        except Exception as e:
            await cl.Message(content=f"❌ Error: {e}").send()
        return

    if low.startswith("/new"):
        new_id = gen_thread_id()
        cl.user_session.set("thread_id", new_id)
        cl.user_session.set("conversation_history", [])
        await cl.Message(content="🔄 New chat started").send()
        return

    if low.startswith("/threads"):
        try:
            threads = get_all_threads() or []
        except Exception as e:
            threads = []
            await cl.Message(content=f"❌ Error getting threads: {e}").send()
        if not threads:
            await cl.Message(content="ℹ️ No saved chats found.").send()
            return
        md_lines = []
        titles = cl.user_session.get("thread_titles") or {}
        for tid in threads[::-1]:
            md_lines.append(f"- **{titles.get(tid,'New Chat')}** — `{tid}`")
        await cl.Message(content="🗂️ **Saved Chats**\n\n" + "\n".join(md_lines)).send()
        return

    if low.startswith("/rename"):
        # format: /rename New Title Here
        parts = user_text.split(" ", 1)
        if len(parts) == 2 and parts[1].strip():
            title = parts[1].strip()
            tid = cl.user_session.get("thread_id")
            titles = cl.user_session.get("thread_titles") or {}
            titles[tid] = title
            cl.user_session.set("thread_titles", titles)
            await cl.Message(content=f"✏️ Renamed chat to **{title}**").send()
        else:
            await cl.Message(content="Usage: `/rename <new title>`").send()
        return

    if low.startswith("/status"):
        rag = get_rag_status()
        if rag.get('has_document'):
            info = rag['document_info']
            await cl.Message(content=(
                f"📚 **Document:** {info['filename']}\n"
                f"📄 Pages: {info['pages']}\n"
                f"🔗 Chunks: {info['chunks']}"
            )).send()
        else:
            await cl.Message(content="ℹ️ No document loaded.").send()
        return

    if low.startswith("/help"):
        await cl.Message(content=(
            "Available commands:\n"
            "• `/upload` — Upload a PDF\n"
            "• `/remove` — Remove current document\n"
            "• `/new` — Start a new chat\n"
            "• `/threads` — List saved chats\n"
            "• `/rename <title>` — Rename current chat\n"
            "• `/status` — Show document status\n\n"
            "Or use the command buttons near the input box."
        )).send()
        return

    # -----------------------
    # Normal chat flow (user query)
    # -----------------------
    # Inform user if a document is loaded
    rag = get_rag_status()
    if rag.get('has_document'):
        docname = rag['document_info']['filename']
        await cl.Message(content=f"📚 Using document: **{docname}**").send()

    # Stream response from backend/chatbot
    await cl.Message(content="⏳ Thinking...").send()

    # Save user message to local conversation_history for convenience (not authoritative)
    conv_hist = cl.user_session.get("conversation_history") or []
    conv_hist.append({"role": "user", "content": user_text})
    cl.user_session.set("conversation_history", conv_hist)

    # Stream chat responses from backend; also detect pending tools afterward
    try:
        full_response = ""
        # chat_stream yields stream events; iterate and send content as they arrive
        async for event in chat_stream(user_text, thread_id):
            # event may be message-like objects; try to extract .content
            content = getattr(event, "content", None)
            if content:
                # stream chunk to user
                await cl.Message(content=content).send()
                full_response += content

        # Save assistant message locally
        conv_hist = cl.user_session.get("conversation_history") or []
        conv_hist.append({"role": "assistant", "content": full_response})
        cl.user_session.set("conversation_history", conv_hist)

        # After response, check for pending tools (HITL)
        pending = await get_pending_tools(thread_id)
        if pending.get('has_pending'):
            cl.user_session.set("awaiting_approval", True)
            cl.user_session.set("pending_tools", pending.get('tools'))
            await show_tool_approval_ui(thread_id, pending.get('tools'))

    except Exception as e:
        await cl.Message(content=f"❌ Error during chat: {e}").send()

# -----------------------
# End of file
# -----------------------
