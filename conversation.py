import re

re_pattern = r'You: [A-Za-z0-9.,;\' ]*'

def read_file(file_name):
    with open('conversation.txt', 'r') as file:
        dialog = file.read()
        return dialog

def get_text(text):
    answers_list = re.findall(re_pattern, text)
    answers_list = [elem.replace('You: ', ' ').strip() for elem in answers_list]
    answers_list.pop()
    return answers_list

t = read_file('conversation.txt')
l = get_text(t)
print(l)
