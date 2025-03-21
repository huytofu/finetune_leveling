import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class Logger:
    """
    A utility class for logging.
    """
    @staticmethod
    def info(message):
        """
        Log an informational message.
        
        Args:
            message (str): The message to log.
        """
        logging.info(message)

    @staticmethod
    def error(message):
        """
        Log an error message.
        
        Args:
            message (str): The message to log.
        """
        logging.error(message)

class ErrorHandler:
    """
    A utility class for error handling.
    """
    @staticmethod
    def handle_error(e, context=""):
        """
        Handle an error by logging it and optionally raising it.
        
        Args:
            e (Exception): The exception to handle.
            context (str): Additional context about where the error occurred.
        """
        Logger.error(f"Error in {context}: {e}")
        raise e 