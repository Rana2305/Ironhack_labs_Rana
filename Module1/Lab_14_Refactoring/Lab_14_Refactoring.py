#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random

age1=int(input("first player please input your age"))
age2=int(input("second player please input your age"))
if age1<age2:
    print("first plyer begins")
else:
    print("second plyer begins")

def roll_dice (num_dice):
    dice_result=[]
    for i in range (0,num_dice):
        d=random.randint(1,6)
        dice_result.append(d)
    return dice_result


remain_number=5
l_m=[]
l6=[]
lif6=[]
while remain_number>0:
    k=0
    result=roll_dice(remain_number)
    for i in range(len(result)):

        if result[i]!=6:
            l6.append(result[i])
            
    
       
    m=max(l6)      
    l_m.append(m)
    remain_number-=1
print(l_m)
total_first_player = sum(l_m)
print(total_first_player)

print("second player")
remain_number=5
l_m=[]
l6=[]
lif6=[]
while remain_number>0:
    k=0
    result=roll_dice(remain_number)
    for i in range(len(result)):
        
        if result[i]!=6:
            l6.append(result[i])
            
        
           
    m=max(l6)      
    l_m.append(m)
    remain_number-=1
print(l_m)
total_second_player = sum(l_m)
print(total_second_player) 

if total_first_player > total_second_player:
    print("first player wins")
elif total_first_player < total_second_player:
    print("second player wins")
else:
    print("there is no winner")

