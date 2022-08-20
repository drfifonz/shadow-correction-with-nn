from datetime import datetime


class Visualizer:
    """
    Visialize model's working
    """

    def __init__(self, opt) -> None:
        self.opt = opt

    def save_images(self, root_path: str, images) -> None:
        """
        saving images in /root_path/actual_date/ folder
        """
        root_folder = root_path + get_current_date_string()

        pass


# helper functions
def get_current_date_string() -> str:
    """
    getting current date and returning it as a string in \n
    YYYY/MM/DD-HH:MM:SS\n
    format
    """
    date = datetime.now()

    day_string = "{:4d}/{:02d}/{:2d}".format(date.year, date.month, date.day)
    hour_string = "{:02d}:{:02d}:{:02d}".format(date.hour, date.minute, date.second)

    return day_string + "-" + hour_string
