#!/usr/bin/env python3
import glob

string_start = 'index.html.html'
string_end = '"'
new_string = 'index.html'

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
