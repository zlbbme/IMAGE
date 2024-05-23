import pyautogui
import pywinauto
import selenium

#获取输入的ID号
def get_id():
    id = input("请输入ID号：")
    return id

#弹出图像框，要求输入ID号
def show_id():
    id = pyautogui.prompt(text='请输入ID号', title='ID号输入框', default='')
    return id


if __name__ == '__main__':
    input_ID = show_id()
    print(input_ID)
