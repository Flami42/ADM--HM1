#PROBLEM1


# INTRODUCTION

##PROBLEM: Say "Hello, World!" With Python
print("Hello, World!")


## PROBLEM: Python If-Else
import math
import os
import random
import re
import sys
if __name__ == '__main__':
    n = int(input().strip())
    if n%2==1:
        print("Weird")
    elif (n%2==0 and n>=2 and n<=5):
        print ("Not Weird")
    elif( n%2==0 and n>=6 and n<=20):
        print("Weird")
    elif(n%2==0 and n>= 20):
        print("Not Weird")


## PROBLEM: Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)


## PROBLEM: Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)



## PROBLEM: Loops
if __name__ == '__main__':
    n = int(input())
    if (n<=20 and n>=1):
        for i in range (n):
            print(i*i)


## PROBLEM: Write a function
def is_leap(year):
    leap = False
    if (year>=1900 and year<=10**5):
        if year%4==0 and (year%100!=0 or year%400==0):
            leap= True
    return leap


## PROBLEM: Print Function
if __name__ == '__main__':
    n=int(input())
    l=str()
    i=1
    while i<=n:
        l=l+str(i)
        i+=1
    print(l)





#BASIC DATA TYPES

## PROBLEM: List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    poss_coords = []
    for i in range (x+1):
        for j in range (y+1):
            for k in range (z+1):
                if i+j+k!=n:
                    poss_coords.append([i, j, k])
    print(poss_coords)

## PROBLEM: Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    if (n>=2 and n<=10):
        arr = list(map(int, input().split(" ")))
        for el in arr:
            if (el>=-100 and el<=100):
                i=True
            else:
                i=False
                break
        if i==True:
            print(sorted(list(set(arr)))[-2])


## PROBLEM: Nested List
if __name__ == '__main__':
    n = int(input())
    if (n>=2 and n<=5):
        d={}
        l=[]
        for _ in range(n):
            name = input()
            score = float(input())
            d[name]=score
        min_val= min(d.values())
        penultimo= sorted(list(set(d.values())))[1]
        for k in d:
            if d[k]== penultimo:
                l.append(k)
        l.sort()
        for el in l:
            print(el)


## PROBLEM: Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    if (n>=2 and n<= 10):
        for _ in range(n):
            name, *line = input().split()
            scores = list(map(float, line))
            student_marks[name] = scores
        query_name = input()
        print(format(sum(student_marks[query_name])/3,".2f"))


## PROBLEM: Lists
if __name__ == '__main__':
    n = int(input())
    lista = []
    for i in range(n):
        cmd= input().split()
        command = cmd[0]
        if command=="print":
            print(lista)
        elif command=="sort":
            lista.sort()
        elif command=="pop":
            lista.pop(len(lista)-1)
        elif command=="reverse":
            lista.reverse()
        elif command=="remove":
            x= int(cmd[1])
            lista.remove(x)
        elif command=="insert":
            x= int(cmd[1])
            y= int(cmd[2])
            lista.insert(x,y)
        elif command=="append":
            x= int(cmd[1])
            lista.append(x)


## PROBLEM: Tuple
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t=tuple(integer_list)
    h=hash(t)
    print(h)


# STRINGS

## PROBLEM: String Validators

if __name__ == '__main__':
    s=input()
    if (len(s)>0 and len(s)<100):
        #print(any(el.isalnum() for el in s))
        #print(any(el.isalpha() for el in s))
        #print(any(el.isdigit() for el in s))
        #print(any(el.islower() for el in s))
        #print(any(el.isupper() for el in s))

        #with the code before it worked, but with 3 test cases it gave me error, so use the eval function
        for method in ('isalnum', 'isalpha', 'isdigit', 'islower', 'isupper'):
            print(any(eval("c." + method + "()") for c in s))


## PROBLEM: sWAP cASE

def swap_case(stringa):
    c=[]
    for a in stringa:
        if a.islower():
            c.append(a.upper())
        elif a.isupper():
            c.append(a.lower())
        else:
            c.append(a)
    c= "".join(c)
    return(c)


## PROBLEM: String Split and Join
def split_and_join(a):
    # write your code here
    a=a.split()
    a="-".join(a)
    return a

## PROBLEM: What's Your Name?
def print_full_name(a, b):
    print("Hello", a,b + "! You just delved into python.")


## PROBLEM: Mutations
def mutate_string(string, position, character):
    position= int(position)
    string=string[:position]+character+string[position+1:]
    return string



## PROBLEM: Find a String
def count_substring(string, sub):
    j=0
    for i in range(0,len(string)+1):
       if string[i: i+len(sub)] == sub:
        j+=1
    return j



## PROBLEM: Text Alignment

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))



## PROBLEM: Text Wrap
def wrap(string, max_width):
    text= textwrap.fill(string,max_width)
    return text


## PROBLEM: Designer Door Mat
n,m= map(int, input().split())
#for all even n
pattern = [('.|.'*(2*i + 1)).center(m, '-') for i in range(n//2)]
print('\n'.join(pattern + ['WELCOME'.center(m, '-')] + pattern[::-1]))


## PROBLEM: String Formatting
def print_formatted(number):
    width=len("{0:b}".format(n))
    for i in range(1,n+1):
        print ("{0:{width}d} {0:{width}o} {0:{width}X} {0:{width}b}".format(i, width=width))


## PROBLEM: Alphabet Rangoli
import string
alpha = string.ascii_lowercase

def print_rangoli(size):
    rangoli = []
    for i in range(n):
        s = "-".join(alpha[i:n])
        rangoli.append((s[::-1]+s[1:]).center(4*n-3, "-"))
    print('\n'.join(rangoli[:0:-1]+rangoli))


## PROBLEM: Capitalize!
def solve(s):
    t=s.split()
    return ' '.join((word.capitalize() for word in t))
    #the code works very good, but the output is not recognised as correct because it is not the same format as the input:
    #(ex: input: "hello   world  lol" return "Hello World Lol" instead of "Hello  World  Lol" )



#SETS


## PROBLEM: No Idea!
n,m=map(int, input().split())
lis= list(map(int, input().split()))
A= list(map(int, input().split()))
B= list(map(int, input().split()))
h=0
if (n,m>=1 and n,m<=10**5):
h=0
for i in range(0, len(lis)):
h += A.count(lis[i])
h -= B.count(lis[i])
print(h)
#the code works, but due to and error ("Your code did not execute within the time limits") ONLY in hackerrank (in other IDE is ok) is not considered valid


## PROBLEM: Introduction to Sets
def average(array):
    arr=set(array)
    av= sum(arr)/len(arr)
    return av


## PROBLEM: Symmetric Difference
m=int(input())
M=set(map(int, input().split()))
n=int(input())
N=set(map(int, input().split()))
for el in sorted(M^N):
    print(el)


## PROBLEM: Set .add()
# Enter your code here. Read input from STDIN. Print output to STDOUT
n= int(input())
A=set()
for i in range(0,n):
    A.add(input())
print(len(A))
## PROBLEM: Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
l = int(input())
for _ in range(l):
    cmd = input().split()
    command = cmd[0]
    if command == "remove" :
        x=int(cmd[1])
        s.remove(x)
    elif command== "pop" :
        s.pop()
    elif command== "discard":
        s.discard(int(cmd[1]))
print(sum(s))


## PROBLEM: Set .union() Operation
n=int(input())
A=set(map(int, input().split()))
m=int(input())
B=set(map(int, input().split()))

print(len(A.union(B)))

## PROBLEM: Set .intersection() Operation
n= int(input())
A= set(map(int, input().split()))
m= int(input())
B= set(map(int, input().split()))
print(len(A.intersection(B)))


## PROBLEM: Set .difference() Operation
n= int(input())
A= set(map(int, input().split()))
m= int(input())
B= set(map(int, input().split()))
print(len(A.difference(B)))
## PROBLEM: Set .symmetric_difference() Operation

n=int(input())
A= set(map(int, input().split()))
m=int(input())
B= set(map(int, input().split()))
print(len(A^B))

## PROBLEM: Set Mutations
n=int(input())
A=set(map(int, input().split()))
N=int(input())
for i in range(N):
    command=input().split()
    other_set=set(map(int, input().split()))
    cmd=command[0]
    len_otherSets= int(command[1])
    if cmd== "intersection_update":
        A.intersection_update(other_set)
    elif cmd=="update":
        A.update(other_set)
    elif cmd== "symmetric_difference_update":
        A.symmetric_difference_update(other_set)
    elif cmd== "difference_update":
        A.difference_update(other_set)
print(sum(A))


## PROBLEM: Check Subset
T=int(input())
if (T>=1 and T<=20):

    for i in range(T):
        a= int(input())
        A= set(map(int, input().split()))
        b= int(input())
        B= set(map(int, input().split()))
        print(A.issubset(B))

## PROBLEM: Check Strict Superset
A=set(map(int, input().split()))
N=int(input())
f=0
for i in range(N):
    otherset=set(map(int, input().split()))
    if (A>otherset):
        f+=1
if f==N:
    print(True)
else:
    print(False)




#COLLECTIONS

##PROBLEM: collections.Counter()

from collections import Counter

x=int(input())
shoes= list(map(int,input().split()))
shoes= Counter(shoes)
money=0
num_costumer= int(input())
for i in range(num_costumer):
    size, price= map(int, input().split())
    if shoes[size]:
        money+=price
        shoes[size]-=1
print(money)


##PROBLEM: DefaultDict Tutorial

from collections import defaultdict

n,m= map(int, input().split())
B =[]
A= defaultdict(list)
for i in range(n):
    word= input()
    A[word].append(i+1)
for i in range(m):
    other_words= input()
    B.append(other_words)

for parole in B:
    if A[parole]:
        print(" ".join(map(str, A[parole])))
    else:
        print("-1")


##PROBLEM: Collections.namedtuple()
from collections import namedtuple
N = int(input())
campi=input().split()
tot=0
for i in range(N):
    Sheet = namedtuple('Sheet', campi)
    x, y,z, w= input().split()
    table= Sheet(x,y,z,w)
    tot+= int(table.MARKS)
print("{:.2f}".format(tot/N))


##PROBLEM: Collections.OrderedDict()

from collections import OrderedDict
spesa = OrderedDict()
n= int(input())
for i in range(n):
    top= input().rpartition(" ")
    price= int(top[-1])
    name= str(top[0])
    spesa[name]= spesa.get(name ,0)+price
for el in spesa:
    print (el, spesa[el])




##PROBLEM: Collections.deque()
from collections import deque
n=int(input())
d=deque()
for i in range(n):
    command= input().split()
    cmd = command[0]
    if cmd== "append":
        d.append(int(command[1]))
    elif cmd == "popleft":
        d.popleft()
    elif cmd== "pop":
        d.pop()
    elif cmd== "appendleft":
        d.appendleft(int(command[1]))
print(" ".join(map(str, d)))

##PROBLEM: Company Logo

import math
import os
import random
import re
import sys

from collections import Counter, OrderedDict

if __name__ == '__main__':
    c= sorted(list(input()))
    c= Counter(c).most_common(3)
    c= OrderedDict(c)
    for el, k in c.items():
        print(el, k)



#DATE AND TIME

##PROBLEM: Calendar Module

import calendar
m, d, y =map(int, input().split())
c = calendar.weekday(y, m, d)
#given the date, the funtion weekday return a number 0-6 which corresponds to a day of the week

if c == 0:
    print("MONDAY")
elif c == 1:
    print("TUESDAY")
elif c == 2:
    print("WEDNESDAY")
elif c==3:
    print("THURSDAY")
elif c==4:
    print("FRIDAY")
elif c== 5:
    print("SATURDAY")
elif c==6:
    print("SUNDAY")



#EXEPTIONS

##PROBLEM: Exeptions

T= int(input())
for i in range(T):
    try:
        a,b= map(int, input().split())
        print(a//b)
    except BaseException as e:
        print("Error Code:",e)



# BUILT-INS

## PROBLEM: Zipped

n, x = map(int, input().split())
grade=[]
for i in range (x): #i create a list of list containing the grades of a subject for each student, the I zip the "matrix" and calculate the average
    grade.append(map(float, input().split()))
zipgrade= zip(grade)
for i in zip(*grade):
    print(sum(i)/x)


##PROBLEM: Athlete Sort

#!/bin/python3

import math
import os
import random
import re
import sys

if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    arr.sort(key=lambda x: x[k])
    for el in arr:
        print(*el, sep=" ")


# PYTHON FUNCTIONALS

##PROBLEM: Map and Lambda Function

cube = lambda x:pow(x,3)  # x**3

def fibonacci(n):
    if n==0:
        lista=[]
        return lista
    elif (n>0 and n<=15):
        lista=[0,1]
        i=2
        while i< n:
            lista.append(lista[-1]+lista[-2])
            i+=1
    # return a list of fibonacci numbers
        return lista[0:n]


 #XML

 ##PROBLEM: XML1
import xml.etree.ElementTree as etree
tree = etree.ElementTree(etree.fromstring(xml))

n=int(input())
code= ""
for i in range(n):
    code+= str(input())
print(code)
#this is what I maneged to do, because I didn"t have time to study the XML Documentation in order to do the exercise

#REGEX AND PARSING

##PROBLEM: Validating Phone numbers
import re
n=int(input())
for _ in range(n):
    phone=input() #phone must be a string to be the argument of the macth function
    #re.match controls that the phone starts with 7 or 8 or 9, and then checks if the rest of the remaining 9 digits are numbers
    if re.match(r'[789]\d{9}$',phone):
        print('YES')
    else:
        print ('NO')



#NUMPY

##PROBLEM: Mean, Var and Std

import numpy
numpy.set_printoptions(sign=' ')
N,M= map(int, input().split())
A=[]
for i in range(N):
    A.append(input().split())
A=numpy.array(A, int)
#numpy.around(arr,decimals=11) #my script works perfectly but due to a hackerrank's bug i have to add this line to make my code "result good" to 2 test cases on 3
print(numpy.mean(A, axis=1))
print(numpy.var(A, axis=0))
print(numpy.around(numpy.std(A),decimals=11))


##PROBLEM:Arrays

def arrays(r):
    r= list(map(float, r))
    r.reverse()
    arr= numpy.array(r, float)
    return arr


##PROBLEM: Shape and Reshape

import numpy
r = input().strip().split()
r= list(map(int, r))
arr= numpy.array(r, int)
print( numpy.reshape(arr,(3,3)))


##PROBLEM: Transpose and Flatten

import numpy
n,m =map(int, input().split())
my_array=[]
for i in range(n):
    my_array.append(input().split())

my_array=numpy.array(my_array, int)
print(numpy.transpose(my_array))
print(my_array.flatten())


##PROBLEM: Concatenate

import numpy
n,m,p= map(int, input().strip().split())
A=[]
B=[]
for i in range(n):
    A.append(input().split())
A= numpy.array(A, int)
for i in range (m):
    B.append(input().split())
B= numpy.array(B, int)
T= numpy.concatenate((A,B), axis= 0)
print(T)


##PROBLEM: Zeros and Ones

import numpy
dim= list(map(int, input().split()))
print(numpy.zeros((dim), dtype = numpy.int))
print(numpy.ones((dim), dtype = numpy.int))


##PROBLEM: Eye and Identity

import numpy
n,m= map(int, input().split())
numpy.set_printoptions(sign=' ') #I took this command from the solution in the discussions because there is a bug with # the test cases (even if the code is correct)
print(numpy.eye(n, m, k =0))


##PROBLEM: Array Mathematics

import numpy
N,M= map(int, input().split())
A=[]
B=[]
for i in range(N):
    A.append(input().split())
A=numpy.array(A, int)
for i in range(N):
    B.append(input().split())
B=numpy.array(B, int)
Met=["+", "-", "*", "//", "%", "**"]
for method in Met:
    print(eval("A" +method+"B"))


##PROBLEM: Floor, Ceil, Rint

import numpy
numpy.set_printoptions(sign=' ')
my_array= numpy.array([*map(float, input().split())])
print(numpy.floor(my_array),numpy.ceil(my_array),numpy.rint(my_array), sep="\n")


##PROBLEM: Sum and Prod

import numpy
N,M= map(int, input().split())
A=[]
for i in range(N):
    A.append(input().split())
A=numpy.array(A, int)
#som= numpy.sum(A, axis=0)
print(numpy.prod(numpy.sum(A, axis=0 )))


##PROBLEM: Min and Max

import numpy
N,M= map(int, input().split())
A=[]
for i in range(N):
    A.append(input().split())
A=numpy.array(A, int)
print(numpy.max(numpy.min(A, axis=1)))


##PROBLEM: Dot and Cross

import numpy
N= int(input())
A=[]
for i in range(N):
    A.append(input().split())
A=numpy.array(A, int)
B=[]
for i in range(N):
    B.append(input().split())
B=numpy.array(B, int)
print(numpy.dot(A,B) )


##PROBLEM: Inner and Outer

import numpy
A=numpy.array([*map(int, input().split())])
B=numpy.array([*map(int, input().split())])
print(numpy.inner(A,B), numpy.outer(A,B), sep="\n")


##PROBLEM: Polynomials

import numpy
arr=numpy.array([*map(float , input().split())])
x=int(input())
print(numpy.polyval(arr, x))


##PROBLEM: Linear Algebra

import numpy
N=int(input())
A=numpy.array([input().split() for _ in range(N)],float)
numpy.set_printoptions(legacy='1.13') #I had to add this line due to an hackerrank's bug
print(numpy.linalg.det(A))

#PROBLEM2

#PROBLEM: Birthday Cake Candles

import math
import os
import random
import re
import sys

def birthdayCakeCandles(ar):
    #the function returns the number of the candles of the maximum high
    return(ar.count(max(ar)))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ar_count = int(input())

    ar = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(ar)

    fptr.write(str(result) + '\n')

    fptr.close()


#PROBLEM:Kangaroo

import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):
    i=0
    y= "YES"
    n="NO"
    while i<10000:  #this cicle controls if both the kangaroo fall into the same "place" after the same samber of jumps
        if (x1+(i*v1) == x2+ (i*v2)):
            return y
            break
        else:
            i+=1
    return n

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()


#PROBLEM:Viral Advertising

import math
import os
import random
import re
import sys

def viralAdvertising(n):
    if (n<=50 and n>1):
        shared=5
        liked=2
        cum=2
        #it keeps the count of everything
        for i in range(1,n):
            shared= liked*3
            liked=(shared//2)
            cum+=liked
        return cum

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()


#PROBLEM: Recursive Digit Sum

import math
import os
import random
import re
import sys

def superDigit(n, k):
    dig=0
    #p is the sum of every numer in the list (n*k)
    p=sum(list(map(int, str(n*k))))
    p=list(map(int, str(p)))
    if (len(p)>1): #if the list consinst in more that one alement, the functions calls himself, else terminate
        som=str(sum(p))
        return superDigit(som,1)
    else:
        return p[0]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()


#PROBLEM 5: INSERTION SORT 1

import math
import os
import random
import re
import sys

def insertionSort1(n, arr):
    v=arr[-1]
    for i in range(n-2, -1,-1):
        if arr[i]> v:
            arr[i+1]=arr[i] #this line shift the number to the right
            print(*arr, sep= " ")
        else:
            arr[i+1] = v
            print(*arr, sep= " ")
            return
    arr[0]=v
    print(*arr, sep= " ")

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)


#PROBLEM 5: INSERTION SORT 2

import math
import os
import random
import re
import sys

def insertionSort2(n, arr):
    for i in range(n):
        t=arr[i]
        j=i #starting from the left these 2 cicli controls that every numeber is in the correct position (ordered)
        while j > 0 and t < arr[j-1]:
            arr[j] = arr[j-1]
            j -= 1
        arr[j] = t
        if i!=0:
            print(*arr, sep= " ")

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)
