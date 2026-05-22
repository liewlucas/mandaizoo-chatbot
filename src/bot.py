import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from src.config import TELEGRAM_BOT_TOKEN
from src.rag import RAGPipeline

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

rag_pipeline = RAGPipeline()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_msg = (
        "Hello! I am the Mandai Zoo Assistant 🦁\n\n"
        "I can answer your questions about the Singapore Zoo and the Mandai Wildlife Reserve "
        "using official information from our website.\n\n"
        "Try asking me:\n"
        "- What are the opening hours?\n"
        "- How do I get there by public transport?\n"
        "- Can I see Orangutans at the zoo?"
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=welcome_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text
    chat_id = update.effective_chat.id
    
    # Show typing indicator
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    
    try:
        # Get answer from RAG pipeline
        answer = rag_pipeline.answer_query(user_query)
        
        try:
            # Try sending with Markdown parsing first
            await context.bot.send_message(chat_id=chat_id, text=answer, parse_mode='Markdown')
        except Exception as parse_error:
            logger.warning(f"Markdown parsing failed, falling back to plain text. Error: {parse_error}")
            # Fallback to plain text if Telegram rejects the markdown
            await context.bot.send_message(chat_id=chat_id, text=answer)
            
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await context.bot.send_message(chat_id=chat_id, text="Sorry, I encountered an error while processing your request. Please try again later.")

def main():
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "your_telegram_bot_token_here":
        logger.error("TELEGRAM_BOT_TOKEN is not configured correctly.")
        return
        
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    logger.info("Starting bot...")
    application.run_polling()

if __name__ == '__main__':
    main()
