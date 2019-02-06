#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
np --version


# In[2]:


print np.__version__


# In[3]:


print(np.__version__)


# In[4]:


np.show_config()


# In[5]:


z=np.zeros(10)


# In[6]:


print(z)


# In[ ]:





# In[7]:


z=np.zeros(10,10)


# In[8]:


z=np.zeros((10,10))


# In[9]:


print("%d bytes", % (z.size))


# In[10]:


print("%d bytes", % (z.size * z.itemsize))


# In[11]:


print("%d bytes", %(z.size * z.itemsize))


# In[12]:


print("%d bytes" % (z.size * z.itemsize))


# In[13]:


print("%d bytes" % (z.size))


# In[14]:


print("%d bytes" % (z.all))


# In[15]:


print("%d bytes" % (z.max))


# In[16]:


get_ipython().run_line_magic('run', '\'python -c "import numpy; numpy.info(numpy.add)"\'')
   


# In[17]:


get_ipython().run_line_magic('run', '\'python3 -c "import numpy; numpy.info(numpy.add)"\'')


# In[18]:


z=np.zeros(10)


# In[19]:


z(4)=1


# In[20]:


z[4]=1


# In[21]:


print(z)


# In[22]:


z=np.arange(10,50)
print(z)


# In[23]:


z=np.arange(50)
z=z[::-1]
print(z)


# In[24]:


z=np.arange(9).reshape(3,3)
print(z)


# In[25]:


print(z.shape)


# In[26]:


#find indices of non zero elements in array
z=np.nonzero([1,2,3,0,2,0,0,22,0])
print(z)


# In[27]:


z=np.eye(2)
print(z)


# In[28]:


z=np.eye(3)


# In[29]:


print(z)


# In[30]:


z=np.random.random(3,3,3)
print(z)


# In[31]:


z=np.random.random((3,3,3))


# In[32]:


print(z)


# In[33]:


z=np.random.random((10,10))
zMin,zMax = z.min(),z.max()
print(zMin)


# In[34]:


print(zMax)


# In[35]:


print(zMin,zmax)


# In[36]:


print(zMin,zMax)


# In[37]:


z=np.random.random(30)
m=z.mean()
print(m)


# In[38]:


z=np.ones((10,10))
z[1:-1,1:-1] = 0
print(z)


# In[39]:


z=np.ones((5,5))
z= np.pad(z, pad_width=1, mode='constant', constant_values=0)
print(z)


# In[40]:


print(0 * np.nan)


# In[41]:


print(np.nan == np.nan)


# In[42]:


print(np.inf > np.nan)


# In[43]:


print(np.nan - np.nan)


# In[44]:


print(np.nan in set([np.nan]))


# In[45]:


print(0.3 == 3 * 0.1)


# In[46]:


print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3 * 0.1)


# In[47]:


z = np.diag(1+np.arange(4),k=-1)
print(z)


# In[48]:


z = np.zeros((8,8),dtype=int)
z[1::2,::2] = 1
z[::2,1::2] = 1
print(z)


# In[49]:


z = np.zeros((8,8))
z[1::2,::2] = 1
z[::2,1::2] = 1
print(z)


# In[50]:


print(np.unravel_index(100,(6,7,8)))


# In[51]:


Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
print(Z)


# In[52]:


z=np.random.random((5,5))
z=(z-np.mean(z))/(np.std(z))
print(z)


# In[53]:




color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])


# In[54]:


print(color)


# In[55]:


z = np.ones((5,3)) @ np.ones((3,2))


# In[56]:


print(z)


# In[57]:


Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1
print(Z)


# In[58]:


print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))


# In[59]:


Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z


# In[60]:


2 << Z >> 2


# In[61]:


Z <- Z


# In[62]:


1j*Z


# In[63]:


Z/1/1


# In[64]:


Z**Z


# In[65]:


print(np.array(0) / np.array(0))
print(np.array(0) // np.array(0))
print(np.array([np.nan]).astype(int).astype(float))


# In[66]:



Z = np.random.uniform(-10,+10,10)
print (np.copysign(np.ceil(np.abs(Z)), Z))


# In[67]:




Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(np.intersect1d(Z1,Z2))


# In[68]:




Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(np.intersect1d(Z1,Z2))


# In[69]:




Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(np.intersect1d(Z1,Z2))


# In[70]:




Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(np.intersect1d(Z1,Z2))


# In[ ]:




