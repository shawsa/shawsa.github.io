#!/usr/bin/env python3
import glob

string_start = '<div class="divSidebar"'
string_end = '</div>'
new_string = '<div class="divSidebar" w3-include-html="../sidebar.html">\n'

target_dir = 'meetings'

if __name__ == '__main__':

    

    for file_path in glob.iglob(target_dir + '**/*.html', recursive=True):
        f = open(file_path, 'r')
        full_html = f.read()
        f.close()

        begin = full_html.find(string_start)
        if begin == -1:
            continue

        end = full_html.find(string_end, begin)

        new_html = full_html[:begin] + new_string + full_html[end:]
        f = open(file_path, 'w')
        f.write(new_html)
        f.close()

        print(file_path)
