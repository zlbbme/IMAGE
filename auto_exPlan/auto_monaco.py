import pyautogui
import pywinauto


def open_plan(patient_id):
    #在当前界面的输入框中输入ID号
    pyautogui.click(100, 100)





if __name__ == '__main__':
    patient_id = input("请输入ID号：")
    open_plan(patient_id)