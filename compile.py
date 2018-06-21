#!/usr/bin/env python3
import argparse, sys
from os import listdir
from os.path import join, getmtime, isfile
import time

import pickle
import io, six, pybtex.database.input.bibtex, pybtex.plugin
from pybtex.database import parse_file

##################################################################################
#
# Extract summary info from article
#
##################################################################################
def read_article(file_path):
    f = open(file_path, 'r')
    full_html = f.read()
    f.close()
    
    title_begin = full_html.find('<span class="title">')
    if title_begin == -1:
        title = 'Missing Title'
    else:
        title_begin += len('<span class="title">')
        title_end = full_html.find("</span>", title_begin)
        title = full_html[title_begin: title_end]

    intro_begin = full_html.find('<p class="intro">')
    if intro_begin == -1:
        intro = 'Missing Intro'
    else:
        intro_begin += len('<p class="intro">')
        intro_end = full_html.find("</p>", intro_begin)
        intro = full_html[intro_begin: intro_end]

    mtime = getmtime(file_path)
    time_string = time.asctime(time.localtime( mtime))

    #intro = intro[:20]

    return mtime, file_path, title, time_string, intro

##################################################################################
#
# Compile the main page
#
##################################################################################
def remove_previous_dir(text):
    text = text.replace('href="../', 'href="')
    return text

def compile_main():
    file_paths = [path for path in listdir('articles') if path[-5:]==".html"]
    articles = [read_article( join('articles', path) ) for path in file_paths]  
    articles.sort(reverse=True) 

    file_paths = [path for path in listdir('experiments') if path[-5:]==".html"]
    experiments = [read_article( join('experiments', path) ) for path in file_paths]  
    experiments.sort(reverse=True) 

    file_paths = [path for path in listdir('meetings') if path[-5:]==".html"]
    meetings = [read_article( join('meetings', path) ) for path in file_paths]  
    meetings.sort(reverse=True, key= lambda x: x[1]) 
    

    # update index
    print('Compiling index.html')
    main_html = '<p class="title_and_author">\n\t<span class="title">Recent Changes</span>\n</p>\n'
    main_html += '<h1>Last Meeting</h1>\n'
    for article in meetings[:1]:
        mtime, file_path, title, time_string, intro = article
        main_html += '<h2><a href="' + file_path + '">' + title + '</a></h2>\n'
        main_html += '<span class="mod_time">' + time_string + '</span>\n'
        main_html += '<p>\n' + remove_previous_dir(intro) + '\n</p>\n'
    main_html += '<h1>Recent Articles</h1>\n'
    for article in articles[:5]:
        mtime, file_path, title, time_string, intro = article
        main_html += '<h2><a href="' + file_path + '">' + title + '</a></h2>\n'
        main_html += '<span class="mod_time">' + time_string + '</span>\n'
        main_html += '<p>\n' + remove_previous_dir(intro) + '\n</p>\n'
    main_html += '<h1>Recent Experiments</h1>\n'
    for experiment in experiments[:5]:
        mtime, file_path, title, time_string, intro = experiment
        main_html += '<h2><a href="' + file_path + '">' + title + '</a></h2>\n'
        main_html += '<span class="mod_time">' + time_string + '</span>\n'
        main_html += '<p>\n' + remove_previous_dir(intro) + '\n</p>\n'

    file_path = 'index.html'
    f = open(file_path, 'r')
    html = f.read()
    f.close()
    
    content_start = html.find('<div class="divContent">') + len('<div class="divContent">')
    content_end = html.find('</div>', content_start)
    new_html = html[:content_start] + main_html + html[content_end:]
    f = open(file_path, 'w')
    f.write(new_html)
    f.close()

    # update recent_articles.html
    print('Compiling recent_articles.html')
    articles_html = '<p class="title_and_author">\n\t<span class="title">Recent Articles</span>\n</p>\n'
    for article in articles[:20]:
        mtime, file_path, title, time_string, intro = article
        articles_html += '<h2><a href="' + file_path + '">' + title + '</a></h2>\n'
        articles_html += '<span class="mod_time">' + time_string + '</span>\n'
        articles_html += '<p>\n' + remove_previous_dir(intro) + '\n</p>\n'
    file_path = 'recent_articles.html'
    f = open(file_path, 'r')
    html = f.read()
    f.close()
    
    content_start = html.find('<div class="divContent">') + len('<div class="divContent">')
    content_end = html.find('</div>', content_start)
    new_html = html[:content_start] + articles_html + html[content_end:]
    f = open(file_path, 'w')
    f.write(new_html)
    f.close()

    # update recent_experiments.html
    print('Compiling recent_experiments.html')
    experiments_html = '<p class="title_and_author">\n\t<span class="title">Recent Experiments</span>\n</p>\n'
    for experiment in experiments[:20]:
        mtime, file_path, title, time_string, intro = experiment
        experiments_html += '<h2><a href="' + file_path + '">' + title + '</a></h2>\n'
        experiments_html += '<span class="mod_time">' + time_string + '</span>\n'
        experiments_html += '<p>\n' + remove_previous_dir(intro) + '\n</p>\n'
    file_path = 'recent_experiments.html'
    f = open(file_path, 'r')
    html = f.read()
    f.close()
    
    content_start = html.find('<div class="divContent">') + len('<div class="divContent">')
    content_end = html.find('</div>', content_start)
    new_html = html[:content_start] + experiments_html + html[content_end:]
    f = open(file_path, 'w')
    f.write(new_html)
    f.close()

    # update last meeting link   
    file_path = 'sidebar.html'
    f = open(file_path, 'r')
    html = f.read()
    f.close()
    link_start = html.find('<a id="last_meeting" href="')
    link_start += len('<a id="last_meeting" href="')
    link_end = html.find('"', link_start)
    meeting_files = [path for path in listdir('meetings') if path[-5:]==".html"]
    meeting_files.sort(reverse=True)
    #check for meeting template
    last_meeting_file = meeting_files[0]
    if last_meeting_file.find('template') > 0 :
        last_meeting_file = meeting_files[1]
    link_html = join('meetings', last_meeting_file)
    new_html = html[:link_start] + link_html + html[link_end:]
    f = open(file_path, 'w')
    f.write(new_html)
    f.close()

    # update sidbar-sub.html
    print('Compiling sidebar-sub.html')
    f = open('sidebar.html', 'r')
    html = f.read()
    f.close()
    new_html = html.replace('href="', 'href="../')
    f = open('sidebar-sub.html', 'w')
    f.write(new_html)
    f.close()

    # update articles.html, experiments.html, meetings.html
    file_paths = ['articles.html', 'experiments.html', 'meetings.html']
    dir_paths = ['articles', 'experiments', 'meetings']
    titles = ['Articles', 'Experiments', 'Meetings']
    content_list = [articles, experiments, meetings]
    for file_path, dir_path, title, contents in zip(file_paths, dir_paths, titles, content_list):
        print('Compiling ' + file_path)
        contents.sort(key= lambda x: x[1])
        if file_path=='meetings.html':
            contents.sort(key= lambda x: x[1], reverse=True)
        articles_html = '<p class="title_and_author">\n\t<span class="title">'
        articles_html += title
        articles_html += '</span>\n</p>\n'
        for article in contents[:]:
            mtime, link_path, title, time_string, intro = article
            link_path
            articles_html += '<h2><a href="' + link_path + '">' + title + '</a></h2>\n'
            articles_html += '<span class="mod_time">' + time_string + '</span>\n'
            articles_html += '<p>\n' + remove_previous_dir(intro) + '\n</p>\n'
        f = open(file_path, 'r')
        html = f.read()
        f.close()
        
        content_start = html.find('<div class="divContent">') + len('<div class="divContent">')
        content_end = html.find('</div>', content_start)
        new_html = html[:content_start] + articles_html + html[content_end:]
        f = open(file_path, 'w')
        f.write(new_html)
        f.close()

##################################################################################
#
# Compile Bibliography Data
#
##################################################################################

def build_bib_dict():
    pybtex_style = pybtex.plugin.find_plugin('pybtex.style.formatting', 'plain')()
    pybtex_html_backend = pybtex.plugin.find_plugin('pybtex.backends', 'html')()
    pybtex_parser = pybtex.database.input.bibtex.Parser()
    
    bib_dict = {}
    bib_files = [bib for bib in listdir('bib') if bib[-4:]=='.bib']
    for bib_file in bib_files:   
        full_path = join('bib', bib_file)
        print(full_path)
        data = parse_file(full_path)
        
        data_formatted = pybtex_style.format_entries(six.itervalues(data.entries))
        output = io.StringIO()
        pybtex_html_backend.write_to_stream(data_formatted, output)
        html = output.getvalue()
        output.close()
        
        html = html.split("<dd>")[1]
        html = html.split("</dd>")[0]
        
        key = next(iter(data.entries))
        bib_dict[key] = html
    
    return bib_dict

def write_full_bib(bib_dict, file_name):
    f = open(file_name, 'w')
    f.write('<html><body><ol>')
    entries = [v for k,v in bib_dict.items()]
    for v in sorted(entries):
        f.write('<li>' + v + '</li>')
    f.close()

##################################################################################
#
# Compile An Article
#
##################################################################################
def complie_article(file_path, bib_dict):
    f = open(file_path, 'r')
    html = f.read()
    f.close()
    
    # updated last modified
    mtime = getmtime(file_path)
    time_string = time.asctime(time.localtime( mtime))
    updated_start = html.find('<span class="updated">') + len('<span class="updated">')
    updated_end = html.find('</span>', updated_start)
    html = html[:updated_start] + time_string + html[updated_end:]

    # create table of contents
    contents_list = []
    contents_entry = []
    h_start = html.find('<div class="divContent">')
    h_start = html.find('<h1', h_start)
    while h_start != -1:
        h_start = html.find('>', h_start) + 1
        h_end = html.find('</h1>', h_start)
        h1 = html[h_start : h_end]
        contents_entry.append(h1)
        next_h_start = html.find('<h1', h_start)
        # collect second headers
        h2_start = html.find('<h2', h_start, next_h_start)
        while h2_start != -1:
            h2_start = html.find('>', h2_start) +1
            h2_end = html.find('</h2>', h2_start)
            h2 = html[h2_start : h2_end]
            contents_entry.append(h2)
            h2_start = html.find('<h2', h2_start, next_h_start)
        contents_list.append(contents_entry)
        contents_entry = []
        h_start = html.find('<h1', h_start+1)
        
    
    contents_html = '\n<ol>\n'
    for section in contents_list:
        contents_html += '\t<li>' + section[0] + '</li>\n'
        if len(section)>1:
            contents_html += '\t<ol>\n'
            for h2 in section[1:]:
                contents_html += '\t\t<li>' + h2 + '</li>\n'
            contents_html += '\t</ol>\n'
    contents_html += '</ol>\n'

    contents_start = html.find('<div class="contents">')
    if contents_start != -1:
        contents_start += len('<div class="contents">')
        contents_end = html.find('</div>', contents_start-1)
        html = html[:contents_start] + contents_html + html[contents_end:]

    # compile citations
    cite_tags = []
    citations = []
    cite_start = html.find('<span class="cite" src="')
    while cite_start != -1:
        cite_start += len('<span class="cite" src="')
        cite_end = html.find('">', cite_start)
        tag = html[cite_start:cite_end]
        cite_start = cite_end +2
        cite_end = html.find('</span>', cite_start)
        
        bib_num = None
        if tag in cite_tags:
            bib_num = cite_tags.index(tag)+1
        elif tag in bib_dict:
            cite_tags.append(tag)
            citations.append(bib_dict[tag])
            bib_num = len(cite_tags)

        if bib_num == None:
            cite_link = '<b>[??]</b>'
        else:
            cite_link = '<b>[<a href="#bib' + str(bib_num) + '">' + str(bib_num) + '</a>]</b>'
            
        html = html[:cite_start] + cite_link + html[cite_end:]
            
        cite_start = html.find('<span class="cite" src="', cite_end)

    # write bibliography
    html_bib = '\n\t<ol>\n'
    for i, citation in enumerate(citations):
        html_bib += '\t\t<li id="bib' + str(i+1) + '">\n\t\t\t'
        html_bib += citation
        html_bib += '\n\t\t</li>\n'
    html_bib += '\t</ol>\n'

    bib_start = html.find('<p class="bibliography">')
    if bib_start != -1:
        bib_start += len('<p class="bibliography">')
        bib_end = html.find('</p>', bib_start)
        html = html[:bib_start] + html_bib + html[bib_end:]

    f = open(file_path, 'w')
    f.write(html)
    f.close()

##################################################################################
#
# main
#
##################################################################################
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compile HTML files.')
    parser.add_argument('files', nargs="*")
    parser.add_argument('--only', '-o', action='store_const', default=False, const=True)
    parser.add_argument('--all', '-a', action='store_const', default=False, const=True)
    parser.add_argument('--experiments', '-e', action='store_const', default=False, const=True)
    parser.add_argument('--meetings', '-m', action='store_const', default=False, const=True)
    parser.add_argument('--bibliography', '-b', action='store_const', default=False, const=True)

    args = parser.parse_args()

    # compile bibliography
    bib_dict = None
    if args.bibliography:
        print('Compiling bibliography...')
        bib_dict = build_bib_dict()
        f = open(join('bin', 'bib.pickle'), 'wb')
        pickle.dump(bib_dict, f)
        f.close()

    # compile articles/experiments/meetings
    compile_dir = 'articles'
    if args.experiments:
        compile_dir = 'experiments'
    elif args.meetings:
        compile_dir = 'meetings'
    
    if len(args.files)>0 or args.all:
        print('Compiling articles...')
        if bib_dict == None:
            bib_dict = pickle.load(open(join('bin', 'bib.pickle'), 'rb'))

        # check if compile all
        if args.all:
            my_files = [my_file[:-5] for my_file in listdir(compile_dir) if my_file[-5:]=='.html']
        else:
            my_files = args.files

        for file_path in my_files:
            path = join(compile_dir, file_path + '.html')
            if isfile(path):
                print('Compiling ' + path)
                complie_article(path, bib_dict)
            else:
                print('Error, ' + path + 'does not exist')

    # compile index.html
    if not args.only:
        compile_main()
        
