def stl(s):
    out = []
    splitted = s.split('.')
    for item in splitted:
        out.append(int(item))
    return out

def lts(l):
    s = ''
    for item in l:
        s += str(item)
        s += '.'
    return s[:-1]

