### Encode basically to encode the amino acid sequence into a [ [Att.1, Att.2...], [Att. 1][Att.2...]] list that can be 
### fed into the neural networkse


f = '''
A 1.8 0 6.0 89.1 0 0 75.8 76.1 1.9
C 2.5 0 5.1 121.2 0 0 115.4 67.9 -1.2
D -3.5 -1 2.8 133.1 0 1 130.3 71.8 -107.3 '''

lines = [x.strip().split() for x in f.strip().split('\n')]

l = len(lines[0]) - 1

### l is the number or length of attributes to each amino acid 

d = {}
for line in lines:
    d[line[0]] = line[1:]

print(d['D']) 

### Format of lines:
### [['Amino Acid', Attribute 1, Att. 2, Att. 3 ...]['Amino Acid', Attribute 1, Att. 2 ...]]

### Format of d:
### {'A': [Att. 1, Att. 2, Att. 3 ...]
###  'D': [Att. 1, Att. 2, Att. 3 ...]}