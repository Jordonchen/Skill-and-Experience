"""
最近复习了操作系统的基本知识，关于页面置换算法的原理，需要在以后多多加强了解。为了更好地理解，打算用python去实现当中一个比较重要的置换算法---LRU。
LRU算法（least recently used）叫做最近最久未使用算法，顾名思义，就是把当一个缺页终端发生的时候，选择最久未被使用的那个页面，并淘汰之。
其依据的是程序的局部性原理，即在近一段时间内，如果某些页面被频繁访问，那么在将来的一段时间内，它们很有可能被再次频繁访问。
反过来说，如果某些页面长时间未被访问，那么在将来一段时间它们还有可能会长时间不被访问。


那么，LRU算法需要记录各个页面使用时间的先后顺序，开销大，不过可能实现的方法如下：
   可以设置一个活动页面栈，当访问某页面时，将此页面压入栈顶，然后考察栈内是否有与此相同的页号，若有，则抽出。
   当需要淘汰一个页面时，总是选择栈底的页面，因为
   它就是最近最久未被使用的页面。
   
   
这边，我举个例子，用python实现LRU算法的场景：
设计一个LRU cache，实现两个功能：(cache中存放着（key,value）键值对)
get(key):获取key对应的value，如果key不在cache中，那么更新；
set(key, value):如果key在cache中则更新它的value；如果不在则插入，如果cache已满则先删除最近最少使用的一项后在插入。
对于get，如果key在cache中，那个get(key）表示了对key的一次访问；而set(key，value)则总是表示对key的一次访问。
使用一个list来记录访问的顺序，最先访问的放在list的前面，最后访问的放在list的后面，故cache已满时，则删除list[0]，然后插入新项；


看起来并不复杂，具体实现，如下：
"""
class LRUcache:
    def __init__(self,capacity):
        self.cache={}
        self.used_list=[]
        self.capacity=capacity
    def get(self,key):
        if key in self.cache:
            if key != self.used_list[-1:]:
                self.used_list.remove(key)
                self.used_list.append(key)
            return self.cache[key]
        else:
            return self.set()
    def set(self,key,value):
        if len(self.cache)==self.capacity:
            self.cache.pop(self.used_list.pop(0))
        self.used_list.append(key)
        self.cache[key]=value
        return value
