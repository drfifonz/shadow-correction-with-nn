import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt

plt.close("all")


class ReadLog:
    def __init__(self, path_to_file) -> None:
        self.path_to_file = path_to_file

        row_dataframe = self.__load_by_pandas(path_to_file)
        self.iter = self.__get_values_from_column(row_dataframe, column_num=0)
        self.loss_G = self.__get_values_from_column(row_dataframe, column_num=1)
        self.loss_G_identity = self.__get_values_from_column(
            row_dataframe, column_num=2
        )
        self.loss_G_GAN = self.__get_values_from_column(row_dataframe, column_num=3)
        self.loss_G_cycle = self.__get_values_from_column(row_dataframe, column_num=4)
        self.loss_D = self.__get_values_from_column(row_dataframe, column_num=5)

    def __load_by_pandas(self, path_to_file: str) -> pd.DataFrame:
        """
        loading dataframe by pandas
        skipping second line
        """
        data = pd.read_table(path_to_file, sep=",", skiprows=[0], header=None)
        # print(type(data))
        # print(data[data.index % 2 == 0])
        return data[data.index % 2 == 0]

    def __get_values_from_column(
        self, row_dataframe: pd.DataFrame, column_num: int
    ) -> np.array:
        """
        geting all values from specific column
        """
        list_of_values = []
        for row in range(row_dataframe.shape[0]):

            cell_data = row_dataframe.iloc[row, column_num]
            cell_data = cell_data.replace("[", "")
            cell_data = cell_data.replace("]", "")

            splitted_cell = cell_data.split()
            cell_value = float(splitted_cell[1])

            list_of_values.append(cell_value)

        return np.array(list_of_values)

    def print_df(self) -> None:
        """
        printing all formatted DataFrame
        """
        print(self.__load_by_pandas(self.path_to_file))

    def plot(self, x: np.array, y: np.array, x_label: str, y_label: str) -> None:
        """
        plotting the dataframe
        """
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)

        combined = pd.concat([x, y], axis=1)
        combined.columns = [f"{x_label}", f"{y_label}"]
        combined.plot(x=f"{x_label}", y=f"{y_label}")
        plt.show()


logs = ReadLog("./data/results/2022-09-16 22:38:45.657194.txt")
logs.print_df()
iterations = logs.iter
loss_G = logs.loss_G
loss_G_identity = logs.loss_G_identity
loss_G_GAN = logs.loss_G_GAN
loss_G_cycle = logs.loss_G_cycle
loss_D = logs.loss_D

logs.plot(
    x=iterations,
    y=loss_D,
    x_label="iterations",
    y_label="discriminator loss",
)
