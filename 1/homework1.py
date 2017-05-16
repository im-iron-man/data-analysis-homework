def classifier(fight, kiss):
    if fight > kiss:
        return 1
    else:
        return 0

x = [[3,104], [2,100], [1,81], [101,10]]
y = [0, 0, 0, 1]

yy = []
for arg in x:
    result = classifier(arg[0], arg[1])
    yy.append(result)
print yy

count = 0
for i in range(len(y)):
    if y[i] != yy[i]:
        count += 1

print 'percentage of mistakes is %f' % (float(count)/len(y))
