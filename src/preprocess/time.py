import datetime as dt
import numpy as np


class TimeManager():
    def __init__(self) -> None:
        pass

    @staticmethod
    def date_2_jd(year, month, day):
        """Return the num of day of year"""
        count, lis = 0, [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0): lis[1] = 29
        for i in range(month - 1):
            count += lis[i]
        return count + day

    def jd(self, begin_date, end_date):
        """Return array of jd index"""
        array = self.get_date_array(begin_date, end_date)
        jd = []
        for date in array:
            jd.append(self.date_2_jd(date.year, date.month, date.day))
        return np.array(jd)

    @staticmethod
    def get_date_array(begin_date, end_date):
        """get array of date according to begin/end date."""

        # Initialize the list from begin_date to end_date
        dates = []

        # Initialize the timeindex for append in dates array.
        _dates = dt.datetime.strptime(begin_date, "%Y-%m-%d")

        # initialized the timeindex for decide whether break loop
        _date = begin_date[:]

        # main loop
        while _date <= end_date:

            # pass date in the array
            dates.append(_dates)

            # refresh date by step 1
            _dates = _dates + dt.timedelta(1)

            # changed condition by step 1
            _date = _dates.strftime("%Y-%m-%d")

        return dates


if __name__ == '__main__':
    print(TimeManager().jd('2002-01-02', '2003-01-02'))