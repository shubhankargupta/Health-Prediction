import pyhs2
import pyhs2
from itertools import chain, combinations
from collections import defaultdict
from sklearn import tree
import io
import pandas as pd
import pygal
from sklearn.cluster import KMeans
import numpy as np
from sklearn import svm
from random import randint


def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
        """calculates the support for items in the itemSet and returns a subset
        of the itemSet each of whose elements satisfies the minimum support"""
        _itemSet = set()
        localSet = defaultdict(int)
        for item in itemSet:
             for transaction in transactionList:
                  if item.issubset(transaction):
                        freqSet[item] += 1
                        localSet[item] += 1
        for item, count in localSet.items():
              support = float(count)/len(transactionList)
              if support >= minSupport:
                      _itemSet.add(item)

        return _itemSet


def joinSet(itemSet, length):
        """Join a set with itself and returns the n-element itemsets"""
        return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


def getItemSetTransactionList(data_iterator):
     transactionList = list()
     itemSet = set()
     for record in data_iterator:
             transaction = frozenset(record)
             transactionList.append(transaction)
     for item in transaction:
              itemSet.add(frozenset([item])) # Generate 1-itemSets
     return itemSet, transactionList

def runApriori(data_iter, minSupport, minConfidence):
        """
        run the apriori algorithm. data_iter is a record iterator
        Return both:
        - items (tuple, support)
        - rules ((pretuple, posttuple), confidence)
        """
        itemSet, transactionList = getItemSetTransactionList(data_iter)
        freqSet = defaultdict(int)
        largeSet = dict()
# Global dictionary which stores (key=n-itemSets,value=support)
# which satisfy minSupport
        assocRules = dict()
# Dictionary which stores Association Rules
        oneCSet = returnItemsWithMinSupport(itemSet,
        transactionList,
        minSupport,
        freqSet)
        currentLSet = oneCSet
        k = 2
        while(currentLSet != set([])):
             largeSet[k-1] = currentLSet
             currentLSet = joinSet(currentLSet, k)
             currentCSet = returnItemsWithMinSupport(currentLSet,
             transactionList,
             minSupport,
             freqSet)
             currentLSet = currentCSet
             k = k + 1


        def getSupport(item):
          """local function which Returns the support of an item"""
          return float(freqSet[item]) / len(transactionList)
        toRetItems = []
        for key, value in largeSet.items():
              toRetItems.extend([(tuple(item), getSupport(item))
                           for item in value])
        toRetRules = []
        for key, value in largeSet.items()[1:]:
                  for item in value:
                     _subsets = map(frozenset, [x for x in subsets(item)])
                     for element in _subsets:
                        remain = item.difference(element)
                        if len(remain) > 0:
                            confidence = getSupport(item) / getSupport(element)
                        if confidence >= minConfidence:
                            toRetRules.append(((tuple(element), tuple(remain)),
                            confidence))
        return toRetItems, toRetRules



def printResults(items, rules,parametera,parameterb):
     c=0
     """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
     for item, support in sorted(items, key=lambda (item, support): support):
             pre=item
             post=item
             if ('TRUE' in str(post) or 'FALSE' in str(post) and (parametera in str(pre) or parametera in str(post)) and (
                 parameterb in str(pre) or parameterb in str(post))):
                 c=1
                 print "item: %s , %.3f" % (str(item), support)
     if(c!=1):
      print("items below minimum support")
     print "\n------------------------ RULES:"
     s=""
     t=""
     con=0
     for rule, confidence in sorted(rules, key=lambda (rule, confidence): confidence):
        pre, post = rule
        #if('TRUE' in str(pre) or 'FALSE' in str(pre) or  'TRUE' in str(post) or 'FALSE' in str(post)):
        if (('TRUE' in str(post) or 'FALSE' in str(post)) and (parametera in str(pre) or parametera in str(post)) and (parameterb in str(pre) or parameterb in str(post)) ):
           if(con<confidence):
              con=confidence
              s=str(pre)
              v=pre
              u=post
              t=str(post)


     if(con!=0):
         print "Rule: %s ==> %s , %.3f" % (s,t, con)
         #print u
         #print v
         e = []
         f = []
         file = open('data', 'r')
         print(s.strip(","))
         for line in file:
             d = []
             u=0
             o=[]
             for p in v:
                 o.insert(0,p)
             for m in o:
                 #print m
                 if m in line:
                     u=1
                 else:
                     u=0

             if  u==1 :
                # print line
                 d.insert(0, line)
                 e.append(d)

         #print e

         e1 = []
         for i in e:

             for j in i:
                 k = (str(j).split(","))
                 c = 1
                 d1 = 1
                 d = []
                 for l in k:
                     # print(l+"l")
                     if c == 1:
                         c += 1
                         d1 = 2

                     elif d1 == 2:
                         d1 += 1
                         y = getclass(l)
                         f.append(y)
                     elif j == '\n':
                         ''''''
                     else:
                         y = gety(l)
                         # print(y+"y")
                         d.insert(0, y)
                 e1.append(d)
         for i in e1:
             i.remove('\n')
         # e1.remove("\n")
         #print e1
         #print f

         with pyhs2.connect(host="localhost",
                            port=10000,
                            authMechanism="PLAIN",
                            user="root",
                            password="cloudera",
                            database=u"minor") as conn:
             g=[]
             e=[]
             f1=[]
             clf = tree.DecisionTreeClassifier(criterion='entropy')
             clf = clf.fit(e1, f)

             with conn.cursor() as cur1:
                 cur1.execute("select * from testingset")
                 for i in cur1.fetch():
                     g.append(i)

                 for i in g:
                     c = 1
                     d1 = 1
                     d = []
                     for j in i:

                         if c == 1:
                             c += 1
                             d1 = 2

                         elif d1 == 2:
                             d1 += 1
                             y = getclass(j)
                             f1.append(y)
                         else:
                             y = gety(j)
                             d.insert(0, y)
                     e.append(d)
                 s = 0
                 tp=0
                 tn=0
                 fn=0
                 fp=0
                 for i in e:
                     t = (clf.predict([i]))
                     if t[0] == 1 and f1[s] == 1:
                         # print("true positive")
                         tp = tp + 1

                     elif t[0] == 1 and f1[s] == 2:
                         # print("false positive")
                         fp = fp + 1
                     elif t[0] == 2 and f1[s] == 1:
                         # print("false negative")
                         fn = fn + 1
                     elif t[0] == 2 and f1[s] == 2:
                         # print("true negative")
                         tn = tn + 1
                     s = s + 1
         pie_chart = pygal.Pie()
         pie_chart.title = 'Results'
         pie_chart.add('TP', tp)
         pie_chart.add('TN', tn)
         pie_chart.add('FP', fp)
         pie_chart.add('FN', fn)

         svg_content = pie_chart.render_in_browser()

         accuracy = (float)(tp + tn) / (float)(tp + tn + fp + fn)
         print("accuracy---", accuracy)
         print("error---- ", (1 - accuracy))
         conn.close()



     else:
            print("items below minimum confidence")



def dataFromFile(fname):
     """Function which reads from the file and yields a generator"""
     file_iter = open(fname, 'rU')
     for line in file_iter:
        line = line.strip().rstrip(',') # Remove trailing comma
        record = frozenset(line.split(','))
        yield record

def apriori():
    with pyhs2.connect(host="localhost",
                       port=10000,
                       authMechanism="PLAIN",
                       user="root",
                       password="cloudera",
                       database=u"minor") as conn:
        file_iter = open('data', 'w')


        with conn.cursor() as cur:
            cur.execute("select * from trainingset")
            for i in cur.fetch():
                for j in i:
                    file_iter.write(str(j)+",")
                file_iter.write("\n")
            file_iter.close()


    inFile = dataFromFile('data')
    parametera=raw_input("first parameter")
    parameterb=raw_input("second parameter")

    minSupport = 0.15
    minConfidence = 0.15
    items, rules = runApriori(inFile, minSupport, minConfidence)
    printResults(items, rules,str(parametera),str(parameterb))

def gety(j):
    if j == 'veg':
        y = 1
    elif j == 'nonveg':
        y = 0
    elif j == 'medium':
        y = 1
    elif j == 'low':
        y = 2
    elif j == 'high':
        y = 3
    elif j == 'black':
        y = 1
    elif j == 'red':
        y = 2
    elif j == 'blue':
        y = 3
    elif j == 'green':
        y = 4
    elif j == 'purple':
        y = 5
    elif j == 'orange':
        y = 6
    elif j == 'yellow':
        y = 7
    elif j == 'white':
        y = 8
    elif j == 'independent':
        y = 1
    elif j == 'kids':
        y = 2
    elif j == 'dependent':
        y = 3
    elif j == 'variety':
        y = 1
    elif j == 'technology':
        y = 2
    elif j == 'none':
        y = 3
    elif j == 'retro':
        y = 4
    elif j == 'ecofriendly':
        y = 5
    elif j == 'thriftyprotector':
        y = 1
    elif j == 'hunterostentatious':
        y = 2
    elif j == 'hardworker':
        y = 3
    elif j == 'conformist':
        y = 4
    elif j == 'Catholic':
        y = 2
    elif j == 'Christian':
        y = 1
    elif j == 'none':
        y = 3
    elif j == 'Mormon':
        y = 4
    elif j == 'Jewish':
        y = 5
    elif j == 'student':
        y = 1
    elif j == 'professional':
        y = 2
    elif j == 'unemployed':
        y = 3
    elif j == 'workingclass':
        y = 4
    elif j == 'abstemious':
        y = 1
    elif j == 'social drinker':
        y = 2
    elif j == 'casual drinker':
        y = 3
    elif j == 'informal':
        y = 1
    elif j == 'formal':
        y = 2
    elif j == 'no preference':
        y = 3
    elif j == 'elegant':
        y = 4
    elif j == 'family':
        y = 1
    elif j == 'friends':
        y = 2
    elif j == 'solitary':
        y = 3
    elif j == 'on foot' or j == 'onfoot':
        y = 1
    elif j == 'public':
        y = 2
    elif j == 'car owner':
        y = 3
    elif j == 'single':
        y = 1
    elif j == 'married':
        y = 2
    elif j == 'widow':
        y = 3
    else:
        y =j
    return y

def getclass(j):
    if j == 'TRUE':
        y = 1
    else:
        y = 2
    return y


def decisiontree(a=1):
 g=[]
 d=[]
 e=[]
 f=[]

 accuracy=0.0
 error=0.0
 tp=0
 fn=0
 fp=0
 tn=0

 with pyhs2.connect(host="localhost",
                   port=10000,
                   authMechanism="PLAIN",
                   user="root",
                   password="cloudera",
                   database=u"minor") as conn:
    with conn.cursor() as cur:
        cur.execute("select * from trainingset")
        for i in cur.fetch():
            g.append(i)

        for i in g:
            c=1
            d1=1
            d=[]
            for j in i:

                if c==1:
                    c+=1
                    d1=2

                elif d1==2:
                    d1+=1
                    y=getclass(j)
                    f.append(y)
                else:
                    y=gety(j)
                    d.insert(0,y)
            e.append(d)
 conn.close()
 if a==1:
   clf = tree.DecisionTreeClassifier(criterion='entropy')
   clf = clf.fit(e, f)
   #print e
   #print f
   e1=[]
   f1=[]
   g1=[]
   d=[]
   with pyhs2.connect(host="localhost",
                      port=10000,
                      authMechanism="PLAIN",
                      user="root",
                      password="cloudera",
                      database=u"minor") as conn:

    with conn.cursor() as cur1:
       cur1.execute("select * from testingset")
       for i in cur1.fetch():
           g1.append(i)

       for i in g1:
           c = 1
           d1 = 1
           d = []
           for j in i:

               if c == 1:
                   c += 1
                   d1 = 2

               elif d1 == 2:
                   d1 += 1
                   y = getclass(j)
                   f1.append(y)
               else:
                   y = gety(j)
                   d.insert(0, y)
           e1.append(d)
       s=0
       for i in e1:
        t = (clf.predict([i]))
        if t[0] == 1 and f[s]==1:
           #print("true positive")
           tp=tp+1

        elif t[0]==1 and f[s]==2:
           #print("false positive")
           fp=fp+1
        elif t[0]==2 and f[s]==1:
           #print("false negative")
           fn=fn+1
        elif t[0]==2 and f[s]==2:
            #print("true negative")
            tn=tn+1
        s=s+1
   pie_chart = pygal.Pie()
   pie_chart.title = 'Results'
   pie_chart.add('TP', tp)
   pie_chart.add('TN', tn)
   pie_chart.add('FP', fp)
   pie_chart.add('FN', fn)


   svg_content = pie_chart.render_in_browser()

   accuracy=(float)(tp+tn)/(float)(tp+tn+fp+fn)
   print("accuracy---",accuracy)
   print("error---- ",(1-accuracy))
   conn.close()
 if a==2:
     clf = tree.DecisionTreeClassifier(criterion='entropy')
     clf = clf.fit(e, f)
     print("enter data")
     food_preference = input("food_preference:1.veg/0.nonveg")
     if food_preference > 1 or food_preference < 0:
         print("error")
         exit()
     # food_preference=gety(food_preference)
     height = input("height in decimal in m")
     if height > 2 or height < 0:
         print("error")
         exit()
     weight = input("enter weight")
     if weight > 150 or weight < 0:
         print("error")
         exit()
     birthyear = input("enter birthyear")
     if birthyear > 2017 or birthyear < 1901:
         print("error")
         exit()
     budget = input("budget:1.medium/2.low/3.high")
     # budget=gety(budget)
     if budget > 3 or budget < 1:
         print("error")
         exit()
     color = input("color:1.black/2.red/3.blue/4.green/5.purple/6.orange/7.yellow/8.white")
     if color > 8 or color < 1:
         print("error")
         exit()

     hijos = input("hijos:1.independent/2.kids/3.dependent")
     if hijos > 3 or hijos < 1:
         print("error")
         exit()

     interest = input("interest:1.variety/2.technology/3.none/4.retro/5.ecofriendly")
     if interest > 5 or interest < 1:
         print("error")
         exit()

     personality = input("personality:1.thriftyprotector/2.hunterostentatious/3.hardworker/4.conformist")
     if personality > 4 or personality < 1:
         print("error")
         exit()

     religion = input("religion:1.christian/2.catholic/3.none/4.mormon/5.jewish")
     if religion > 5 or religion < 1:
         print("error")
         exit()

     activity = input("activity:1.student/2.professional/3.unmployed/4.workingclass")
     if activity > 4 or activity < 1:
         print("error")
         exit()

     status = input("status:1.single/2.married/3.widow")
     if status > 3 or status < 1:
         print("error")
         exit()

     transport = input("transport:1.onfoot/2.public/3.carowner")
     if transport > 3 or transport < 1:
         print("error")
         exit()

     ambience = input("ambience:1.family/2.friends/3.solitary")
     if ambience > 3 or ambience < 1:
         print("error")
         exit()

     dresspref = input("dresspref:1.informal/2.formal/3.nopreference/4.elegant")
     if dresspref > 4 or dresspref < 1:
         print("error")
         exit()

     drinklevel = input("drinklevel:1.abstemious/2.social/3.casual")
     if drinklevel > 3 or drinklevel < 1:
         print("error")
         exit()

     t=(clf.predict([[food_preference,height,budget,weight,color,activity,religion,personality,interest,birthyear,hijos,status,transport,
        ambience,dresspref,drinklevel]]))
     if t[0]==1:
         print("Healthy")
     else:
         print("Not Healthy")
 if a==3:
     clf = tree.DecisionTreeClassifier(criterion='gini')
     clf = clf.fit(e, f)
     # print e
     # print f
     e1 = []
     f1 = []
     g1 = []
     d = []
     with pyhs2.connect(host="localhost",
                        port=10000,
                        authMechanism="PLAIN",
                        user="root",
                        password="cloudera",
                        database=u"minor") as conn:

         with conn.cursor() as cur1:
             cur1.execute("select * from testingset")
             for i in cur1.fetch():
                 g1.append(i)

             for i in g1:
                 c = 1
                 d1 = 1
                 d = []
                 for j in i:

                     if c == 1:
                         c += 1
                         d1 = 2

                     elif d1 == 2:
                         d1 += 1
                         y = getclass(j)
                         f1.append(y)
                     else:
                         y = gety(j)
                         d.insert(0, y)
                 e1.append(d)
             s = 0
             for i in e1:
                 t = (clf.predict([i]))
                 if t[0] == 1 and f[s] == 1:
                     # print("true positive")
                     tp = tp + 1

                 elif t[0] == 1 and f[s] == 2:
                     # print("false positive")
                     fp = fp + 1
                 elif t[0] == 2 and f[s] == 1:
                     # print("false negative")
                     fn = fn + 1
                 elif t[0] == 2 and f[s] == 2:
                     # print("true negative")
                     tn = tn + 1
                 s = s + 1
     pie_chart = pygal.Pie()
     pie_chart.title = 'Results'
     pie_chart.add('TP', tp)
     pie_chart.add('TN', tn)
     pie_chart.add('FP', fp)
     pie_chart.add('FN', fn)

     svg_content = pie_chart.render_in_browser()

     accuracy = (float)(tp + tn) / (float)(tp + tn + fp + fn)
     print("accuracy---", accuracy)
     print("error---- ", (1 - accuracy))
     conn.close()
 if a==4:
     clf = tree.DecisionTreeClassifier(criterion='gini')
     clf = clf.fit(e, f)
     print("enter data")
     food_preference = input("food_preference:1.veg/0.nonveg")
     if food_preference > 1 or food_preference < 0:
         print("error")
         exit()
     # food_preference=gety(food_preference)
     height = input("height in decimal in m")
     if height > 2 or height < 0:
         print("error")
         exit()
     weight = input("enter weight")
     if weight > 150 or weight < 0:
         print("error")
         exit()
     birthyear = input("enter birthyear")
     if birthyear > 2017 or birthyear < 1901:
         print("error")
         exit()
     budget = input("budget:1.medium/2.low/3.high")
     # budget=gety(budget)
     if budget > 3 or budget < 1:
         print("error")
         exit()
     color = input("color:1.black/2.red/3.blue/4.green/5.purple/6.orange/7.yellow/8.white")
     if color > 8 or color < 1:
         print("error")
         exit()

     hijos = input("hijos:1.independent/2.kids/3.dependent")
     if hijos > 3 or hijos < 1:
         print("error")
         exit()

     interest = input("interest:1.variety/2.technology/3.none/4.retro/5.ecofriendly")
     if interest > 5 or interest < 1:
         print("error")
         exit()

     personality = input("personality:1.thriftyprotector/2.hunterostentatious/3.hardworker/4.conformist")
     if personality > 4 or personality < 1:
         print("error")
         exit()

     religion = input("religion:1.christian/2.catholic/3.none/4.mormon/5.jewish")
     if religion > 5 or religion < 1:
         print("error")
         exit()

     activity = input("activity:1.student/2.professional/3.unmployed/4.workingclass")
     if activity > 4 or activity < 1:
         print("error")
         exit()

     status = input("status:1.single/2.married/3.widow")
     if status > 3 or status < 1:
         print("error")
         exit()

     transport = input("transport:1.onfoot/2.public/3.carowner")
     if transport > 3 or transport < 1:
         print("error")
         exit()

     ambience = input("ambience:1.family/2.friends/3.solitary")
     if ambience > 3 or ambience < 1:
         print("error")
         exit()

     dresspref = input("dresspref:1.informal/2.formal/3.nopreference/4.elegant")
     if dresspref > 4 or dresspref < 1:
         print("error")
         exit()

     drinklevel = input("drinklevel:1.abstemious/2.social/3.casual")
     if drinklevel > 3 or drinklevel < 1:
         print("error")
         exit()

     t=(clf.predict([[food_preference, height, budget, weight, color, activity, religion, personality, interest,
                         birthyear, hijos, status, transport,
                         ambience, dresspref, drinklevel]]))
     if t[0]==1:
         print("Healthy")
     else:
         print("Not Healthy")
def kmeansfun(a1=1):
    g = []
    d = []
    e = []
    f = []

    accuracy = 0.0
    error = 0.0

    tp = 0
    fn = 0
    fp = 0
    tn = 0

    with pyhs2.connect(host="localhost",
                       port=10000,
                       authMechanism="PLAIN",
                       user="root",
                       password="cloudera",
                       database=u"minor") as conn:
        with conn.cursor() as cur:
            cur.execute("select * from trainingset")
            for i in cur.fetch():
                g.append(i)

            for i in g:
                c = 1
                d1 = 1
                d = []
                for j in i:

                    if c == 1:
                        c += 1
                        d1 = 2

                    elif d1 == 2:
                        d1 += 1
                        y = getclass(j)
                        f.append(y)
                    else:
                        y = gety(j)
                        d.insert(0, y)
                e.append(d)
    conn.close()

    if a1==1:
      x = np.array(e)
      kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
      #print(kmeans.labels_)
      p=0
      r1=[]
      s1=[]
      r2=[]
      s2=[]
      for h in kmeans.labels_:
          if h==0:
              r1.append(e[p])
              s1.append(f[p])
              p=p+1
          else:
              r2.append(e[p])
              s2.append(f[p])
              p=p+1
      #print r1
      #print s1
      clf = tree.DecisionTreeClassifier(criterion='gini')
      r=randint(0,1)
      if r==0:
       clf = clf.fit(r1,s1)
      elif r==1:
        clf = clf.fit(r2,s2)
      #kmeans.labels_ = np.array(f)
      e1 = []
      f1 = []
      g1 = []
      d = []
      with pyhs2.connect(host="localhost",
                         port=10000,
                         authMechanism="PLAIN",
                         user="root",
                         password="cloudera",
                         database=u"minor") as conn:

          with conn.cursor() as cur1:
              cur1.execute("select * from testingset")
              for i in cur1.fetch():
                  g1.append(i)

              for i in g1:
                  c = 1
                  d1 = 1
                  d = []
                  for j in i:

                      if c == 1:
                          c += 1
                          d1 = 2

                      elif d1 == 2:
                          d1 += 1
                          y = getclass(j)
                          f1.append(y)
                      else:
                          y = gety(j)
                          d.insert(0, y)
                  e1.append(d)
              s = 0
              for i in e1:
                  t = (clf.predict([i]))
                  if t[0] == 1 and f[s] == 1:
                      # print("true positive")
                      tp = tp + 1

                  elif t[0] == 1 and f[s] == 2:
                      # print("false positive")
                      fp = fp + 1
                  elif t[0] == 2 and f[s] == 1:
                      # print("false negative")
                      fn = fn + 1
                  elif t[0] == 2 and f[s] == 2:
                      # print("true negative")
                      tn = tn + 1
                  s = s + 1
      pie_chart = pygal.Pie()
      pie_chart.title = 'Results'
      pie_chart.add('TP', tp)
      pie_chart.add('TN', tn)
      pie_chart.add('FP', fp)
      pie_chart.add('FN', fn)

      svg_content = pie_chart.render_in_browser()

      accuracy = (float)(tp + tn) / (float)(tp + tn + fp + fn)
      print("accuracy---", accuracy)
      print("error---- ", (1 - accuracy))
      conn.close()

    if a1 == 2 :
        x = np.array(e)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
        p = 0
        r1 = []
        s1 = []
        r2 = []
        s2 = []
        for h in kmeans.labels_:
            if h == 0:
                r1.append(e[p])
                s1.append(f[p])
                p = p + 1
            else:
                r2.append(e[p])
                s2.append(f[p])
                p = p + 1
        # print r1
        # print s1
        clf = tree.DecisionTreeClassifier(criterion='gini')
        r = randint(0, 1)
        if r == 0:
            clf = clf.fit(r1, s1)
        elif r == 1:
            clf = clf.fit(r2, s2)

        print("enter data")
        food_preference = input("food_preference:1.veg/0.nonveg")
        if food_preference > 1 or food_preference < 0:
            print("error")
            exit()
        # food_preference=gety(food_preference)
        height = input("height in decimal in m")
        if height > 2 or height < 0:
            print("error")
            exit()
        weight = input("enter weight")
        if weight > 150 or weight < 0:
            print("error")
            exit()
        birthyear = input("enter birthyear")
        if birthyear > 2017 or birthyear < 1901:
            print("error")
            exit()
        budget = input("budget:1.medium/2.low/3.high")
        # budget=gety(budget)
        if budget > 3 or budget < 1:
            print("error")
            exit()
        color = input("color:1.black/2.red/3.blue/4.green/5.purple/6.orange/7.yellow/8.white")
        if color > 8 or color < 1:
            print("error")
            exit()

        hijos = input("hijos:1.independent/2.kids/3.dependent")
        if hijos > 3 or hijos < 1:
            print("error")
            exit()

        interest = input("interest:1.variety/2.technology/3.none/4.retro/5.ecofriendly")
        if interest > 5 or interest < 1:
            print("error")
            exit()

        personality = input("personality:1.thriftyprotector/2.hunterostentatious/3.hardworker/4.conformist")
        if personality > 4 or personality < 1:
            print("error")
            exit()

        religion = input("religion:1.christian/2.catholic/3.none/4.mormon/5.jewish")
        if religion > 5 or religion < 1:
            print("error")
            exit()

        activity = input("activity:1.student/2.professional/3.unmployed/4.workingclass")
        if activity > 4 or activity < 1:
            print("error")
            exit()

        status = input("status:1.single/2.married/3.widow")
        if status > 3 or status < 1:
            print("error")
            exit()

        transport = input("transport:1.onfoot/2.public/3.carowner")
        if transport > 3 or transport < 1:
            print("error")
            exit()

        ambience = input("ambience:1.family/2.friends/3.solitary")
        if ambience > 3 or ambience < 1:
            print("error")
            exit()

        dresspref = input("dresspref:1.informal/2.formal/3.nopreference/4.elegant")
        if dresspref > 4 or dresspref < 1:
            print("error")
            exit()

        drinklevel = input("drinklevel:1.abstemious/2.social/3.casual")
        if drinklevel > 3 or drinklevel < 1:
            print("error")
            exit()

        t = (clf.predict([[food_preference, height, budget, weight, color, activity, religion, personality, interest,
                           birthyear, hijos, status, transport,
                           ambience, dresspref, drinklevel]]))
        if t[0] == 1:
            print("Healthy")
        else:
            print("Not Healthy")

def svmfun(a1=1):
    g = []
    d = []
    e = []
    f = []

    accuracy = 0.0
    error = 0.0

    tp = 0
    fn = 0
    fp = 0
    tn = 0

    with pyhs2.connect(host="localhost",
                       port=10000,
                       authMechanism="PLAIN",
                       user="root",
                       password="cloudera",
                       database=u"minor") as conn:
        with conn.cursor() as cur:
            cur.execute("select * from trainingset")
            for i in cur.fetch():
                g.append(i)

            for i in g:
                c = 1
                d1 = 1
                d = []
                for j in i:

                    if c == 1:
                        c += 1
                        d1 = 2

                    elif d1 == 2:
                        d1 += 1
                        y = getclass(j)
                        f.append(y)
                    else:
                        y = gety(j)
                        d.insert(0, y)
                e.append(d)
    conn.close()

    if a1==1:

      clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
      clf.fit(e,f)
      e1 = []
      f1 = []
      g1 = []
      d = []
      with pyhs2.connect(host="localhost",
                         port=10000,
                         authMechanism="PLAIN",
                         user="root",
                         password="cloudera",
                         database=u"minor") as conn:

          with conn.cursor() as cur1:
              cur1.execute("select * from testingset")
              for i in cur1.fetch():
                  g1.append(i)

              for i in g1:
                  c = 1
                  d1 = 1
                  d = []
                  for j in i:

                      if c == 1:
                          c += 1
                          d1 = 2

                      elif d1 == 2:
                          d1 += 1
                          y = getclass(j)
                          f1.append(y)
                      else:
                          y = gety(j)
                          d.insert(0, y)
                  e1.append(d)
              s = 0
              for i in e1:
                  t = (clf.predict([i]))
                  if t[0] == 1 and f[s] == 1:
                      # print("true positive")
                      tp = tp + 1

                  elif t[0] == 1 and f[s] == 2:
                      # print("false positive")
                      fp = fp + 1
                  elif t[0] == 2 and f[s] == 1:
                      # print("false negative")
                      fn = fn + 1
                  elif t[0] == 2 and f[s] == 2:
                      # print("true negative")
                      tn = tn + 1
                  s = s + 1
      pie_chart = pygal.Pie()
      pie_chart.title = 'Results'
      pie_chart.add('TP', tp)
      pie_chart.add('TN', tn)
      pie_chart.add('FP', fp)
      pie_chart.add('FN', fn)

      svg_content = pie_chart.render_in_browser()

      accuracy = (float)(tp + tn) / (float)(tp + tn + fp + fn)
      print("accuracy---", accuracy)
      print("error---- ", (1 - accuracy))
      conn.close()

    if a1 == 2 :

        clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
                      max_iter=-1, probability=False, random_state=None, shrinking=True,
                      tol=0.001, verbose=False)
        clf.fit(e, f)

        print("enter data")
        food_preference = input("food_preference:1.veg/0.nonveg")
        if food_preference >1 or food_preference<0:
            print("error")
            exit()
        # food_preference=gety(food_preference)
        height = input("height in decimal in m")
        if height > 2 or height <0:
            print("error")
            exit()
        weight = input("enter weight")
        if weight >150 or weight<0:
            print("error")
            exit()
        birthyear = input("enter birthyear")
        if birthyear >2017 or birthyear<1901:
            print("error")
            exit()
        budget = input("budget:1.medium/2.low/3.high")
        # budget=gety(budget)
        if budget >3 or budget<1:
            print("error")
            exit()
        color = input("color:1.black/2.red/3.blue/4.green/5.purple/6.orange/7.yellow/8.white")
        if color > 8 or color < 1:
            print("error")
            exit()

        hijos = input("hijos:1.independent/2.kids/3.dependent")
        if hijos > 3 or hijos < 1:
            print("error")
            exit()

        interest = input("interest:1.variety/2.technology/3.none/4.retro/5.ecofriendly")
        if interest > 5 or interest < 1:
            print("error")
            exit()

        personality = input("personality:1.thriftyprotector/2.hunterostentatious/3.hardworker/4.conformist")
        if personality > 4 or personality < 1:
            print("error")
            exit()

        religion = input("religion:1.christian/2.catholic/3.none/4.mormon/5.jewish")
        if religion > 5 or religion < 1:
            print("error")
            exit()

        activity = input("activity:1.student/2.professional/3.unmployed/4.workingclass")
        if activity > 4 or activity < 1:
            print("error")
            exit()

        status = input("status:1.single/2.married/3.widow")
        if status > 3 or status < 1:
            print("error")
            exit()


        transport = input("transport:1.onfoot/2.public/3.carowner")
        if transport > 3 or transport < 1:
            print("error")
            exit()

        ambience = input("ambience:1.family/2.friends/3.solitary")
        if ambience > 3 or ambience < 1:
            print("error")
            exit()

        dresspref = input("dresspref:1.informal/2.formal/3.nopreference/4.elegant")
        if dresspref > 4 or dresspref < 1:
            print("error")
            exit()

        drinklevel = input("drinklevel:1.abstemious/2.social/3.casual")
        if drinklevel > 3 or drinklevel < 1:
            print("error")
            exit()


        t = (clf.predict([[food_preference, height, budget, weight, color, activity, religion, personality, interest,
                           birthyear, hijos, status, transport,
                           ambience, dresspref, drinklevel]]))
        if t[0] == 1:
            print("Healthy")
        else:
            print("Not Healthy")




if __name__== "__main__":

    choice=input("enter your choice-\n1.Decision tree(entropy) \n2.Predict by decision tree(entropy)\n3.Decision tree(gini) "
                 "\n4.Predict by decision"
                 "tree(gini)\n5.Kmeans\n6.Predict by Kmeans \n7.SVM \n8.Predict by SVM\n9.Apriori" )
    if choice==1:
        decisiontree()
    elif choice==2:
        decisiontree(2)
    elif choice==3:
        decisiontree(3)
    elif choice==4:
        decisiontree(4)
    elif choice==5:
        kmeansfun()
    elif choice==6:
        kmeansfun(2)
    elif choice==7:
        svmfun()
    elif choice==8:
        svmfun(2 )
    elif choice==9:
        apriori()
    else:
        print("wrong choice")
        exit()






