def latex_table(t, headers=None, row_labels=None, format_str='%.5g'):
    cols = len(t[0])
    rows = len(t)
    if not headers is None:
        assert len(headers) == cols
    if not row_labels is None:
        assert len(row_labels) == rows
    
    ret = '\\begin{table}\n\t\\begin{center}\n'
    if row_labels is None:
        ret += '\t\\begin{tabular}{' + '|c'*cols + '|}\n'
    else:
        ret += '\t\\begin{tabular}{' + '|c'*(cols+1) + '|}\n'
    ret += '\t\t\\hline\n'
    if not headers is None:
        header_str = ''
        if not row_labels is None:
            header_str += '&'
        for h in headers:
            header_str += str(h) + '&'
        header_str = header_str[:-1] #remove trailing &
        ret+= '\t\t' + header_str + '\\\\ \\hline\n'
        
    for i in range(rows):
        col = ''
        if not row_labels is None:
            col += row_labels[i] + '&'
        for j in range(cols):
            if type(t[i][j]) is str:
                col += t[i][j] + '&'
            else:
                col += (format_str % t[i][j]) + '&' #str(t[i,j]) + '&'
        col = col[:-1] #remove trailing &
        ret+= '\t\t' + col + '\\\\ \\hline \n'
    ret+= '\t\\end{tabular}\n\t\\end{center}\n\\end{table}\n'
    return ret

def html_table(t, headers=None, row_labels=None, format_str='%.5g'):
    cols = len(t[0])
    rows = len(t)
    if not headers is None:
        assert len(headers) == cols
    if not row_labels is None:
        assert len(row_labels) == rows
    
    ret = '<table>\n'
    if not headers is None:
        header_str = '\t<tr>\n'
        if not row_labels is None:
            header_str += '\t\t<td></td>\n'
        for h in headers:
            header_str += '\t\t<td>' + str(h) + '</td>\n'
        ret+= header_str + '\t</tr>\n'
        
    for i in range(rows):
        col = '\t<tr>\n'
        if not row_labels is None:
            col += '\t\t<td>' + row_labels[i] + '</td>\n'
        for j in range(cols):
            if type(t[i,j]) is str:
                col += '\t\t<td>' + t[i,j] + '</td>\n'
            else:
                col += '\t\t<td>' + (format_str % t[i,j]) + '</td>\n'
        ret+= col + '\t</tr>\n'
    ret+= '</table>'
    return ret

def text_table(t, headers=None, row_labels=None, format_str='%.5g', width=15):
    cols = len(t[0])
    rows = len(t)
    if not headers is None:
        assert len(headers) == cols
    if not row_labels is None:
        assert len(row_labels) == rows
    
    ret = ''
    if not headers is None:
        header_str = ''
        if not row_labels is None:
            header_str += ' '*width
        for h in headers:
            header_str += str(h).ljust(width)
        ret+= header_str + '\n'
        
    for i in range(rows):
        col = ''
        if not row_labels is None:
            col += row_labels[i].ljust(width)
        for j in range(cols):
            if type(t[i,j]) is str:
                col += t[i,j].ljust(width)
            else:
                col += (format_str % t[i,j]).ljust(width)
        ret+= col + '\n'
    return ret


