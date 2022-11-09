import pygame
import math
import keyboard as k
import numpy
import time
import random
import sys

class Proccessor:
    def __init__(self,size=100,minPitch=144,maxSemitones=36,delay=0.001,duration=0.005,sample_rate=44100,bits=16,font="verdana",fontsize=18,exitkey="esc",channels=20,vol=0.1):
        pygame.mixer.pre_init(sample_rate,-bits, 2)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(channels)
        self.vol = vol
        self.exitkey = exitkey
        self.pygameInit()
        self.semitone = 2**(1/12)
        self.bits = bits
        self.sample_rate = sample_rate
        self.duration = duration
        self.delay = delay
        self.font = pygame.font.SysFont(font,fontsize)
        self.fontsize = fontsize
        self.size = size
        self.memory = list(range(1,size+1))
        self.drawstep = (self.width-20)//size
        self.pixelSpread = -(-self.size//(self.width-20-size*self.drawstep))
        self.comparisons = 0
        self.swaps = 0
        self.accesses = 0
        self.highlighted = dict()
        self.minPitch = minPitch
        self.maxSemitones = maxSemitones
        self.action = "Sorting Visualizer"
        self.display()

    def __repr__(self):
        return repr(self.memory)

    def escape(self):
        if k.is_pressed(self.exitkey):
            pygame.quit()
            sys.exit()

    def pygameInit(self):
        pygame.init()

        pygame.display.set_caption("Sorting Visualizer")

        resolution = pygame.display.Info()
        self.width = resolution.current_w
        self.height = resolution.current_h
        self.hwidth = (self.width/2)
        self.hheight = (self.height/2)

        self.screen = pygame.display.set_mode((self.width,self.height),pygame.NOFRAME)

    def maprange(self,val,a,b,c,d):
        return ((val-a)/(b-a))*(d-c)+c

    def squareWave(self,x,freq):
        a = 1/freq
        b = x%(a)
        a /= 2
        return (1 if b<a else -1)

    def makeSound(self,freq,freq2=None):
        if freq2 == None: freq2 = freq

        n_samples = round(self.duration*self.sample_rate)

        buf = numpy.zeros((n_samples, 2), dtype = numpy.int16)
        max_sample = 2**(self.bits-1) - 1

        for s in range(n_samples):
            t = s/self.sample_rate
            buf[s][0] = (max_sample*self.squareWave(t,freq)*self.vol)
            buf[s][1] = (max_sample*self.squareWave(t,freq2)*self.vol)

        sound = pygame.sndarray.make_sound(buf)
        pygame.mixer.find_channel(True).play(sound)
        time.sleep(self.delay)

    def changeSize(self,newsize):
        self.size = newsize
        self.memory = list(range(1,newsize+1))
        self.drawstep = (self.width-20)//newsize
        self.pixelSpread = -(-self.size//(self.width-20-newsize*self.drawstep))
        self.display()

    def resetStats(self):
        self.comparisons = 0
        self.swaps = 0
        self.accesses = 0
        self.action = ""

    def displayinfo(self):
        self.screen.blit(self.font.render(self.action,1,(255,255,255)),(5,5))
        self.screen.blit(self.font.render(str(self.size)+" Numbers",1,(255,255,255)),(5,self.fontsize+5))
        self.screen.blit(self.font.render("Delay: "+str(self.delay*100)+"ms",1,(255,255,255)),(5,self.fontsize*2+5))
        self.screen.blit(self.font.render("Comparisons: "+str(self.comparisons),1,(255,255,255)),(5,self.fontsize*3+5))
        self.screen.blit(self.font.render("Swaps: "+str(self.swaps),1,(255,255,255)),(5,self.fontsize*4+5))
        self.screen.blit(self.font.render("Accesses: "+str(self.accesses),1,(255,255,255)),(5,self.fontsize*5+5))

    def display(self):
        pygame.event.pump()
        self.escape()
        self.screen.fill((0,0,0))
        x = 10
        for i in range(self.size):
            val = self.memory[i]
            hei = round(self.maprange(val,0,self.size,1,self.height-self.fontsize*7))
            col = (255,255,255)
            for colour in self.highlighted:
                if i in self.highlighted[colour]:
                    col = colour[0]
            pygame.draw.rect(self.screen,col,(x,self.height-hei,self.drawstep+(i%self.pixelSpread==0),hei))
            x += self.drawstep+(i%self.pixelSpread==0)
        self.displayinfo()
        self.highlighted = {colour:idxes for colour,idxes in self.highlighted.items() if colour[1] == 1}
        pygame.display.update()

    def sleep(self,seconds):
        start = time.time()
        while time.time()-start < seconds:
            pygame.event.pump()
            self.escape()

    def highlight(self,*idxs,col,permanent=False):
        if permanent:
            if (col,1) in self.highlighted: self.highlighted[(col,1)].update(idxs)
            else: self.highlighted[(col,1)] = set(idxs)
        else:
            self.highlighted[(col,0)] = idxs

    def removeHighlight(self,col=None,permanent=False):
        if col == None: self.highlighted = dict()
        else:
            try: del self.highlighted[(col,permanent)]
            except: pass

    def convToFreq(self,val):
        return self.minPitch*(self.semitone**self.maprange(val,1,self.size,0,self.maxSemitones))

    def play(self,idx):
        val = self.memory[idx]
        self.highlight(idx,col=(255,0,0))
        self.display()
        self.makeSound(self.convToFreq(val))

    def access(self,idx):
        self.accesses += 1
        val = self.memory[idx]
        self.highlight(idx,col=(255,0,0))
        self.display()
        self.makeSound(self.convToFreq(val))
        return val

    def define(self,idx,val):
        self.accesses += 1
        self.memory[idx] = val
        self.highlight(idx,col=(255,0,0))
        self.display()
        self.makeSound(self.convToFreq(val))
        return val

    def swap(self,idx1,idx2):
        self.swaps += 1
        self.accesses += 2
        temp = self.memory[idx1]
        self.memory[idx1] = self.memory[idx2]
        self.memory[idx2] = temp
        self.highlight(idx1,idx2,col=(0,255,0))
        self.display()
        self.makeSound(self.convToFreq(self.memory[idx1]),self.convToFreq(self.memory[idx2]))

    def compare(self,idx1,idx2):
        self.comparisons += 1
        self.accesses += 2
        a = self.memory[idx1]
        b = self.memory[idx2]
        self.highlight(idx1,idx2,col=(0,0,255))
        self.display()
        self.makeSound(self.convToFreq(self.memory[idx1]),self.convToFreq(self.memory[idx2]))
        return (-1 if a<b else (1 if a>b else 0))

    def insert(self,idx1,idx2):
        if idx1 < idx2:
            for i in range(idx1,idx2):
                _.swap(i,i+1)
        else:
            for i in range(idx1,idx2,-1):
                _.swap(i,i-1)

    def shuffle(self):
        self.delay = 0.1/self.size
        self.resetStats()
        self.action = "Shuffling..."
        for i in range(self.size-1): self.swap(i,random.randint(i+1,self.size-1))
        self.swap(self.size-1,random.randint(0,self.size-2))

        self.removeHighlight()
        self.display()

    def reverse(self):
        for i in range(self.size//2):
            self.swap(i,self.size-1-i)

    def done(self):
        self.removeHighlight()
        
        self.delay = 0.5/self.size
        step = max(1,self.size//100)
        for i in range(0,self.size,step):
            self.highlight(*range(i,i+step),col=(0,255,0),permanent=True)
            self.play(i)
            if self.memory[i] != i+1: raise Exception("Not Sorted!")

        self.removeHighlight((255,0,0))

        self.display()

        self.removeHighlight()
        self.resetStats()
        self.sleep(2)

    def kill(self):
        pygame.quit()
        sys.exit()

class Algorithm:
    def __init__(self,name,func,numbers,*args,delay=None):
        self.name = name
        self.numbers = numbers
        self.func = func
        self.args = args
        if delay == None: self.delay = 0.1/numbers
        else: self.delay = delay

    def run(self,_):
        _.changeSize(algorithm.numbers)
        _.shuffle()
        _.sleep(1)
        _.resetStats()
        _.delay = self.delay
        _.action = algorithm.name
        self.func(_,*self.args)
        _.done()

def algorithms():
    def bubbleSort(_):
        for i in range(_.size):
            done = 1
            for j in range(_.size-1-i):
                if _.compare(j,j+1) == 1:
                    _.swap(j,j+1)
                    done = 0
            if done: break

    def cocktailShakerSort(_):
        n = _.size
        done = 0
        start = 0
        end = n-1
        while 1:
            done = 1
            for i in range(start,end):
                if _.compare(i,i+1) == 1:
                    _.swap(i,i+1)
                    done = 0
            if done: break
            done = 1
            end  -= 1
            for i in range(end-1, start-1, -1):
                if _.compare(i,i+1) == 1:
                    _.swap(i,i+1)
                    done = 0
            start += 1
            if done: break

    def gnomeSort(_):
        idx = 0
        while idx < _.size:
            if idx == 0: idx += 1
            if _.compare(idx,idx-1) == -1:
                _.swap(idx,idx-1)
                idx -= 1
            else: idx += 1

    def oddEvenSort(_):
        notdone = 1
        while notdone:
            notdone = 0
            for i in range(1,_.size-1,2):
                if _.compare(i,i+1) == 1:
                    _.swap(i,i+1)
                    notdone = 1
                      
            for i in range(0,_.size-1, 2):
                if _.compare(i,i+1) == 1:
                    _.swap(i,i+1)
                    notdone = 1

    def selectionSort(_):
        for i in range(_.size):
            min_idx = i
            for j in range(i+1,_.size):
                if _.compare(min_idx,j) == 1:
                    min_idx = j

            _.swap(i,min_idx)

    def doubleSelectionSort(_):
        n = _.size
        i = 0
        j = n-1
        while(i < j):
            min_idx = i
            max_idx = i
            for k in range(i, j + 1, 1):
                if _.compare(k,max_idx) == 1:
                    max_idx = k
                elif _.compare(k,min_idx) == -1:
                    min_idx = k
             
            _.swap(i,min_idx)
            
            if _.compare(i,max_idx) == 0:
                _.swap(j,min_idx)
            else:
                _.swap(j,max_idx)
     
            i += 1
            j -= 1

    def insertionSort(_):
        for i in range(1,_.size):
            j = i-1
            while 1:
                if j<0: break
                if _.compare(j+1,j)==-1:
                    _.swap(j+1,j)
                    j -= 1
                else: break

    def binary_search(_,idx,l,r):
        if l==r:
            if _.compare(l,idx)==1: return l
            else: return l+1

        if l>r: return l
      
        m = (l+r)//2
        result = _.compare(m,idx)
        if result == -1: return binary_search(_,idx,m+1,r)
        elif result == 1: return binary_search(_,idx,l,m-1)
        else: return m
      
      
    def binaryInsertionSort(_):
        for i in range(1,_.size):
            j = binary_search(_,i,0,i-1)
            _.insert(i,j)

    def nextGapForCombSort(gap):
        gap = (gap * 10)//13
        return max(1,gap)

    def combSort(_):
        n = _.size
        gap = n
        swapped = 1
        while gap !=1 or swapped == 1:
            gap = nextGapForCombSort(gap)
            swapped = 0
            for i in range(0, n-gap):
                if _.compare(i,i+gap) == 1:
                    _.swap(i,i+gap)
                    swapped = 1

    def partition(_,l,h):
        i = l-1
      
        for j in range(l, h):
            if _.compare(j,h)<1:
                i += 1
                _.swap(i,j)

        _.swap(i+1,h)
        return i+1

    def recursiveQuickSort(_,l=0,h=None):
        if h == None: h = _.size-1
        
        if l < h:
            pi = partition(_,l,h)
            recursiveQuickSort(_, l,pi-1)
            recursiveQuickSort(_,pi+1,h)

    def iterativeQuickSort(_):
        size = _.size
        stack = [0]*size
        top = 1
        stack[1] = size-1

        while top >= 0:
            h = stack[top]
            top -= 1
            l = stack[top]
            top -= 1

            p = partition(_,l,h)

            if p-1 > l:
                top += 1
                stack[top] = l
                top += 1
                stack[top] = p-1
      
            if p+1 < h:
                top += 1
                stack[top] = p+1
                top += 1
                stack[top] = h

    def merge(_,l,m,r):
        a = l
        b = l
        c = m+1

        n = _.size

        temp = [*_.memory]
     
        while b <= m and c <= r:
            if _.compare(b,c)==-1:
                temp[a] = _.access(b)
                b += 1
            else:
                temp[a] = _.access(c)
                c += 1
            a += 1
     
        while b < n and b <= m:
            temp[a] = _.access(b)
            a += 1
            b += 1
     
        for b in range(l,r+1):
            _.define(b,temp[b])

    def recursiveMergeSort(_,l=0,r=None):
        if r == None: r = _.size-1
        
        if l == r: return
        m = (l+r)//2
        recursiveMergeSort(_,l,m)
        recursiveMergeSort(_,m+1,r)
        merge(_,l,m,r)

    def iterativeMergeSort(_):
        width = 1
        n = _.size
        while width < n:
            l=0
            while (l < n): 
                r = min(l+(width*2-1), n-1)
                m = min(l+width-1,n-1)
                merge(_,l,m,r)
                l += width*2
            width *= 2

    def maxHeapify(_,n,i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2

        if l < n and _.compare(largest,l) == -1: largest = l
        if r < n and _.compare(largest,r) == -1: largest = r

        if largest != i:
            _.swap(i,largest)
            maxHeapify(_,n,largest)

    def recursiveMaxHeapSort(_):
        n = _.size

        for i in range((n//2)-1,-1,-1):
            maxHeapify(_,n,i)

        for i in range(n-1,0,-1):
            _.swap(i,0)
            maxHeapify(_,i,0)

    def minHeapify(_,n,i):
        smallest = i
        l = 2 * i + 1
        r = 2 * i + 2

        if l < n and _.compare(l,smallest) == -1: smallest = l
        if r < n and _.compare(r,smallest) == -1: smallest = r
     
        if smallest != i:
            _.swap(i,smallest)
            minHeapify(_,n,smallest)

    def recursiveMinHeapSort(_):
        n = _.size

        for i in range((n//2)-1,-1,-1):
            minHeapify(_,n,i)

        for i in range(n-1,0,-1):
            _.swap(i,0)
            minHeapify(_,i,0)

        _.reverse()

    def buildMaxHeap(_):
        for i in range(_.size):
            if _.compare(i,int((i-1)/2)) == 1:
                j = i
                while _.compare(j,int((j-1)/2)) == 1:
                    _.swap(j,int((j-1)/2))
                    j = int((j-1)/2)
     
    def iterativeMaxHeapSort(_):
        buildMaxHeap(_)
     
        for i in range(_.size-1,0,-1):
            _.swap(0,i)
         
            j = 0
            idx = 0

            while 1:
                idx = 2*j+1
                if idx<(i-1) and _.compare(idx,idx+1) == -1: idx += 1
                if idx<i and _.compare(j,idx) == -1: _.swap(j,idx)
                j = idx
                if idx >= i: break

    def buildMinHeap(_):
        for i in range(_.size):
            if _.compare(i,int((i-1)/2)) == -1:
                j = i
                while _.compare(j,int((j-1)/2)) == -1:
                    _.swap(j,int((j-1)/2))
                    j = int((j-1)/2)

    def iterativeMinHeapSort(_):
        buildMinHeap(_)
     
        for i in range(_.size-1,0,-1):
            _.swap(0,i)
         
            j = 0
            idx = 0

            while 1:
                idx = 2*j+1
                if idx<(i-1) and _.compare(idx,idx+1) == 1: idx += 1
                if idx<i and _.compare(j,idx) == 1: _.swap(j,idx)
                j = idx
                if idx >= i: break

        _.reverse()

    def countingSort(_):
        n = _.size
        
        countarray = [0 for i in range(n+1)]
        output = [0] * n
        
        for i in range(n): countarray[_.access(i)] += 1
        for i in range(1,len(countarray)): countarray[i]+=countarray[i-1]
        for i in range(n):
            ith = _.access(i)
            output[countarray[ith] - 1] = ith
            countarray[ith] -= 1

        for i in range(n):
            _.define(i,output[i])

    def nToBaseb(n,b):
        if n == 0: return [0]
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return digits

    def countingSortForLSDRadixSort(_,converted,base,digit):
        n = _.size
        output = [0] * n
        base10output = [0] * n
        count = [0] * base

        for i in range(n):
            try: radix = converted[i][digit]
            except: radix = 0
            count[radix] += 1
        for i in range(1,base): count[i] += count[i-1]

        i = n-1
        while i >= 0:
            try: radix = converted[i][digit]
            except: radix = 0
            output[count[radix]-1] = converted[i]
            base10output[count[radix]-1] = _.access(i)
            count[radix] -= 1
            i -= 1

        for i in range(n):
            converted[i] = output[i]
            _.define(i,base10output[i])

    def radixSortLSD(_,b):
        converted = []
        for i in range(_.size): converted.append(nToBaseb(_.access(i),b))
        for i in range(math.ceil(math.log(_.size,b))):
            countingSortForLSDRadixSort(_,converted,b,i)

    def countingSortForMSDRadixSort(_,l,r,converted,base,digit):
        if digit == -1 or l>r: return
        n = (r-l+1)
        output = [0] * n
        base10output = [0] * n
        count = [0] * base

        for i in range(l,r+1):
            try: radix = converted[i][digit]
            except: radix = 0
            count[radix] += 1
        for i in range(1,base): count[i] += count[i-1]

        copy = [*count]

        i = r
        while i >= 0:
            try: radix = converted[i][digit]
            except: radix = 0
            output[count[radix]-1] = converted[i]
            base10output[count[radix]-1] = _.access(i)
            count[radix] -= 1
            i -= 1

        for i in range(l,r+1):
            converted[i] = output[i-l]
            _.define(i,base10output[i-l])

        l_ = l
        for i in range(base):
            countingSortForMSDRadixSort(_,l_,l_+copy[i]-1,converted,base,digit-1)
            l_ += copy[i]

    def radixSortMSD(_,b):
        raise NotImplementedError
        idx = 0
        for i in range(_.size):
            if _.compare(i,idx) == 1: idx = i
        
        converted = []
        for i in range(_.size): converted.append(nToBaseb(_.access(i),b))
        countingSortForMSDRadixSort(_,0,_.size-1,converted,b,len(converted[idx])-1)
        
        
    yield Algorithm("Bubble Sort",bubbleSort,100)
    yield Algorithm("Cocktail Shaker Sort",cocktailShakerSort,100)
    yield Algorithm("Gnome Sort",gnomeSort,100)
    yield Algorithm("Odd Even Sort",oddEvenSort,100)
    yield Algorithm("Selection Sort",selectionSort,100)
    yield Algorithm("Double Selection Sort",doubleSelectionSort,100)
    yield Algorithm("Insertion Sort",insertionSort,100)
    yield Algorithm("Binary Insertion Sort",binaryInsertionSort,100)
    yield Algorithm("Comb Sort",combSort,200)
    yield Algorithm("Quick Sort (Recursive)",recursiveQuickSort,400)
    yield Algorithm("Quick Sort (Iterative)",iterativeQuickSort,400)
    yield Algorithm("Merge Sort (Recursive)",recursiveMergeSort,400)
    yield Algorithm("Merge Sort (Iterative)",iterativeMergeSort,400)
    yield Algorithm("In Place Merge Sort",inPlaceMergeSort,400)        
    yield Algorithm("Max Heap Sort (Recursive)",recursiveMaxHeapSort,400)
    yield Algorithm("Max Heap Sort (Iterative)",iterativeMaxHeapSort,400)
    yield Algorithm("Min Heap Sort (Recursive)",recursiveMinHeapSort,400)
    yield Algorithm("Min Heap Sort (Iterative)",iterativeMinHeapSort,400)
    yield Algorithm("Counting Sort",countingSort,400)
    yield Algorithm("Radix Sort LSD (Base 10)",radixSortLSD,400,10)
    #yield Algorithm("Radix Sort MSD (Base 10)",radixSortMSD,400,10)


if __name__=="__main__":
    _ = Proccessor()
    _.sleep(3)

    for algorithm in algorithms(): algorithm.run(_)

    _.kill()
