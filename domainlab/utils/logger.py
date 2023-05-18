import os
import logging


class Logger:
    logger = None

    @staticmethod
    def get_logger(logger_name='logger', loglevel='DEBUG'):
        if Logger.logger is None:
            Logger.logger = logging.getLogger(logger_name)
            Logger.logger.setLevel(loglevel)

            # Create handlers and set their logging level
            logfolder = 'zoutput/logs'
            os.makedirs(logfolder, exist_ok=True)
            # Create handlers and set their logging level
            filehandler = logging.FileHandler(logfolder + '/' + Logger.logger.name + '.log',
                                              mode='w')
            filehandler.setLevel('DEBUG')
            # Add handlers to logger
            Logger.logger.addHandler(filehandler)
        return Logger.logger
