import logging

def set_log_file(filename):
    logging.basicConfig(
        filename=filename,
        filemode="w",
        format="%(asctime)s,%(msecs)d %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
