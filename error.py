# -*- coding: utf-8 -*-
import math


class Errors:
    
    def __init__(self, real_samples, gen_samples):
        self.real_samples=real_samples
        self.gen_samples=gen_samples
        self.ntest=len(real_samples[0])
        self.num_columns=len(real_samples)#number of stations(6)

    def cumulative(self, i, s):
        summ=0
        xi=self.gen_samples[s][i]
        for x in self.real_samples[s]:
            if x<=xi:
                summ+=1
        return (summ+1)/(self.ntest+2)        
    
    def marginal(self):
        w=[0, 0, 0, 0, 0, 0]
        for s in range(self.num_columns):
            for i in range(self.ntest):
                u=self.cumulative(i, s)
                u1=self.cumulative(self.ntest-i-1, s)
                w[s]+=(2*i+1)*(math.log(u)+math.log(1-u1))/(-self.ntest)
            w[s] -= self.ntest
        return sum(w)/self.num_columns  
   
    def r(self, i, b):
        summ=0
        if b==0:
            data=self.gen_samples
        else:
            data=self.real_samples
        for j in range(self.ntest):
            if not(j==i):
                ind=True
                for s in range(self.num_columns):
                    if data[s][j]>=data[s][i]:
                        ind=False
                if ind:
                    summ+=1
        return summ/(self.ntest-1)                
    
    def dependency(self):
        summ=0
        for i in range(self.ntest):
            summ+=abs(self.r(i, 0)-self.r(i, 1))
        return summ/self.ntest