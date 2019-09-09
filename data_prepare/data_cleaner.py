import numpy as np
import pandas as pd
from data_prepare.data_importer import DataImporter

# raw data constants
STUDENTID = "STUDENTID"
ACCESSION_NUMBER = "AccessionNumber"
ITEM_TYPE = "ItemType"
OBSERVABLE = "Observable"
EVENT_TIME = "EventTime"
CLICK_PROGRESS_NAVIGATOR = "Click Progress Navigator"
BLOCK_REV = "BlockRev"
EOS_TIME_LFT = "EOSTimeLft"
HELP_MAT_8 = "HELPMAT8"
SEC_TIME_OUT = "SecTimeOut"
VERB_CHAIN = "verb_chain"
EXTENDED_INFO = "ExtendedInfo"
# Observable constants
CALCULATOR_BUFFER = "Calculator Buffer"
CLICK_CHOICE = "Click Choice"
OPEN_CALCULATOR = "Open Calculator"
CLOSE_CALCULATOR = "Close Calculator"
DROP_CHOICE = "DropChoice"
ELIMINATE_CHOICE = "Eliminate Choice"
EQUATION_EDITOR_BUTTON = "Equation Editor Button"
HORIZONTAL_ITEM_SCROLL = "Horizontal Item Scroll"
VERTICAL_ITEM_SCROLL = "Vertical Item Scroll"
MATH_KEYPRESS = "Math Keypress"
MOVE_CALCULATOR = "Move Calculator"
TEXT_TO_SPEECH = "TextToSpeech"
YES = "Yes"
NO = "No"
# select columns
SELECT_COLUMNS = [STUDENTID, ACCESSION_NUMBER, ITEM_TYPE, OBSERVABLE, EVENT_TIME]


class DataCleaner(object):
    @classmethod
    def clean_df(cls, df):
        df = df.pipe(ExtendedInfoCleaner.clean_extended_info).pipe(
            ColumnCleaner.clean_columns
        )
        return df


class ExtendedInfoCleaner(object):
    @classmethod
    def clean_extended_info(cls, df):
        df = (
            df.pipe(cls.clean_calculator_buffer)
            .pipe(cls.clean_click_choice)
            .pipe(cls.clean_drop_choice)
            .pipe(cls.clean_close_open_calculator)
            .pipe(cls.clear_eliminate_choice)
            .pipe(cls.clean_equation_editor_button)
            .pipe(cls.clean_item_scroll)
            .pipe(cls.clean_math_key_press)
            .pipe(cls.clean_move_calculator)
            .pipe(cls.clean_text_to_speech)
            .pipe(cls.clean_yes_no)
            .pipe(cls.clean_na)
        )
        return df

    @classmethod
    def clean_calculator_buffer(cls, df):
        df.loc[df[OBSERVABLE] == CALCULATOR_BUFFER, EXTENDED_INFO] = np.nan
        return df

    @classmethod
    def clean_click_choice(cls, df):
        """
        统计选择的选项
        """
        tmp_df = df.loc[df[OBSERVABLE] == CLICK_CHOICE, EXTENDED_INFO].str.extract(
            r"^.*?_(\d):.*$"
        )
        df.loc[df[OBSERVABLE] == CLICK_CHOICE, EXTENDED_INFO] = tmp_df.values
        return df

    @classmethod
    def clean_close_open_calculator(cls, df):
        for verb in [OPEN_CALCULATOR, CLOSE_CALCULATOR]:
            df.loc[df[OBSERVABLE] == verb, EXTENDED_INFO] = np.nan
        return df

    @classmethod
    def clean_drop_choice(cls, df):
        """
        统计drop中list长度
        """
        tmp_df = df.loc[df[OBSERVABLE] == DROP_CHOICE, EXTENDED_INFO].str.count(
            "source"
        )
        df.loc[df[OBSERVABLE] == DROP_CHOICE, EXTENDED_INFO] = tmp_df.values
        return df

    @classmethod
    def clear_eliminate_choice(cls, df):
        tmp_df = df.loc[df[OBSERVABLE] == ELIMINATE_CHOICE, EXTENDED_INFO].str.extract(
            r"^.*?_(\d:.*$)"
        )
        df.loc[df[OBSERVABLE] == ELIMINATE_CHOICE, EXTENDED_INFO] = tmp_df.values
        return df

    @classmethod
    def clean_equation_editor_button(cls, df):
        df.loc[df[OBSERVABLE] == EQUATION_EDITOR_BUTTON, EXTENDED_INFO] = np.nan
        return df

    @classmethod
    def clean_item_scroll(cls, df):
        for scroll in [HORIZONTAL_ITEM_SCROLL, VERTICAL_ITEM_SCROLL]:
            sub_df = df.loc[df[OBSERVABLE] == scroll, EXTENDED_INFO]
            tmp_df = (
                sub_df.str.extract(r"(^\d*)?,.*$")
                + "_"
                + sub_df.str.extract(r"^\d*?,(.*?),.*$")
            )
            df.loc[df[OBSERVABLE] == scroll, EXTENDED_INFO] = tmp_df.values
        return df

    @classmethod
    def clean_math_key_press(cls, df):
        df.loc[df[OBSERVABLE] == MATH_KEYPRESS, EXTENDED_INFO] = np.nan
        return df

    @classmethod
    def clean_move_calculator(cls, df):
        df.loc[df[OBSERVABLE] == MOVE_CALCULATOR, EXTENDED_INFO] = np.nan
        return df

    @classmethod
    def clean_text_to_speech(cls, df):
        df.loc[df[OBSERVABLE] == TEXT_TO_SPEECH, EXTENDED_INFO] = np.nan
        return df

    @classmethod
    def clean_yes_no(cls, df):
        for verb in [YES, NO]:
            df.loc[df[OBSERVABLE] == verb, EXTENDED_INFO] = np.nan
        return df

    @classmethod
    def clean_na(cls, df):
        df.loc[:, EXTENDED_INFO] = df.loc[:, EXTENDED_INFO].fillna("").astype(str)
        return df


class ColumnCleaner(object):
    @classmethod
    def clean_columns(cls, df):
        df = df.pipe(cls.combine_verb).pipe(cls.change_time_type).pipe(cls.drop_columns)
        return df

    @classmethod
    def combine_verb(cls, df):
        df.loc[:, OBSERVABLE] = df[OBSERVABLE] + "_" + df[EXTENDED_INFO]
        return df

    @classmethod
    def change_time_type(cls, df):
        df.loc[:, "EventTime"] = pd.to_datetime(df.EventTime)
        df = df.query(f"{EVENT_TIME}.notnull()")
        return df

    @classmethod
    def drop_columns(cls, df):
        df = df[SELECT_COLUMNS]
        return df


if __name__ == "__main__":
    test_data = DataImporter().data_a_train
    x = DataCleaner.clean_df(test_data)
    print("x")
