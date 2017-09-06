# -*- coding: utf-8 -*-


import os, sys
import pickle #store and load variables
# import shelve #store and load variables



#############################
#       PROGRESS BAR        #
#############################

def progress(count, total, status=''):
# As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 



#############################
#       STORING DATA        #
#############################

    
# def store(obj, target):
#     my_shelf = shelve.open(target,'n') # 'n' for new
#     my_shelf[target] = obj
#     my_shelf.close()


# def restore(source):
#     my_shelf = shelve.open(source)
#     data = my_shelf[source]
#     my_shelf.close()
#     return data


def store(obj, target):
    file = open(target, 'wb')
    pickle.Pickler(file).dump(obj)
    # pickle.dump(obj, f)
    file.close()

def restore(source):
    f = open(source, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj