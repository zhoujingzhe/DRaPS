import logging.config

logging.config.fileConfig('loggin.conf')
logger = logging.getLogger('simpleExample')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('Console.log')
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.info('Starting to training')

