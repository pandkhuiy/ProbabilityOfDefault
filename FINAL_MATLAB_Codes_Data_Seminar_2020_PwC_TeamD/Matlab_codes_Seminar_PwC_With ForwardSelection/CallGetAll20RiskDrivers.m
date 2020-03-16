%% This code computes the predictor matrices with all 20 features for each quarter of each year: 2000Q1 till 2018Q4 
load('Yvectors12months.mat'); % This is obtained with the getDefaultVector.m function

1
[X_WoE2000Q1,InfoValue2000Q1,y2000Q1]  = GetAll20RiskDrivers('Acquisition_2000Q1.txt',y2000Q1);
2
[X_WoE2000Q2,InfoValue2000Q2,y2000Q2]  = GetAll20RiskDrivers('Acquisition_2000Q2.txt',y2000Q2);
3
[X_WoE2000Q3,InfoValue2000Q3,y2000Q3]  = GetAll20RiskDrivers('Acquisition_2000Q3.txt',y2000Q3);
4
[X_WoE2000Q4,InfoValue2000Q4,y2000Q4]  = GetAll20RiskDrivers('Acquisition_2000Q4.txt',y2000Q4);
5
[X_WoE2001Q1,InfoValue2001Q1,y2001Q1]  = GetAll20RiskDrivers('Acquisition_2001Q1.txt',y2001Q1);
6
[X_WoE2001Q2,InfoValue2001Q2,y2001Q2]  = GetAll20RiskDrivers('Acquisition_2001Q2.txt',y2001Q2);
7
[X_WoE2001Q3,InfoValue2001Q3,y2001Q3]  = GetAll20RiskDrivers('Acquisition_2001Q3.txt',y2001Q3);
8
[X_WoE2001Q4,InfoValue2001Q4,y2001Q4]  = GetAll20RiskDrivers('Acquisition_2001Q4.txt',y2001Q4);
9
[X_WoE2002Q1,InfoValue2002Q1,y2002Q1]  = GetAll20RiskDrivers('Acquisition_2002Q1.txt',y2002Q1);
10
[X_WoE2002Q2,InfoValue2002Q2,y2002Q2]  = GetAll20RiskDrivers('Acquisition_2002Q2.txt',y2002Q2);
11
[X_WoE2002Q3,InfoValue2002Q3,y2002Q3]  = GetAll20RiskDrivers('Acquisition_2002Q3.txt',y2002Q3);
12
[X_WoE2002Q4,InfoValue2002Q4,y2002Q4]  = GetAll20RiskDrivers('Acquisition_2002Q4.txt',y2002Q4);
13
[X_WoE2003Q1,InfoValue2003Q1,y2003Q1]  = GetAll20RiskDrivers('Acquisition_2003Q1.txt',y2003Q1);
14
[X_WoE2003Q2,InfoValue2003Q2,y2003Q2]  = GetAll20RiskDrivers('Acquisition_2003Q2.txt',y2003Q2);
15
[X_WoE2003Q3,InfoValue2003Q3,y2003Q3]  = GetAll20RiskDrivers('Acquisition_2003Q3.txt',y2003Q3);
16
[X_WoE2003Q4,InfoValue2003Q4,y2003Q4]  = GetAll20RiskDrivers('Acquisition_2003Q4.txt',y2003Q4);
17
[X_WoE2004Q1,InfoValue2004Q1,y2004Q1]  = GetAll20RiskDrivers('Acquisition_2004Q1.txt',y2004Q1);
18
[X_WoE2004Q2,InfoValue2004Q2,y2004Q2]  = GetAll20RiskDrivers('Acquisition_2004Q2.txt',y2004Q2);
19
[X_WoE2004Q3,InfoValue2004Q3,y2004Q3]  = GetAll20RiskDrivers('Acquisition_2004Q3.txt',y2004Q3);
20
[X_WoE2004Q4,InfoValue2004Q4,y2004Q4]  = GetAll20RiskDrivers('Acquisition_2004Q4.txt',y2004Q4);
21
[X_WoE2005Q1,InfoValue2005Q1,y2005Q1]  = GetAll20RiskDrivers('Acquisition_2005Q1.txt',y2005Q1);
22
[X_WoE2005Q2,InfoValue2005Q2,y2005Q2]  = GetAll20RiskDrivers('Acquisition_2005Q2.txt',y2005Q2);
23
[X_WoE2005Q3,InfoValue2005Q3,y2005Q3]  = GetAll20RiskDrivers('Acquisition_2005Q3.txt',y2005Q3);
24
[X_WoE2005Q4,InfoValue2005Q4,y2005Q4]  = GetAll20RiskDrivers('Acquisition_2005Q4.txt',y2005Q4);
25
[X_WoE2006Q1,InfoValue2006Q1,y2006Q1]  = GetAll20RiskDrivers('Acquisition_2006Q1.txt',y2006Q1);
26
[X_WoE2006Q2,InfoValue2006Q2,y2006Q2]  = GetAll20RiskDrivers('Acquisition_2006Q2.txt',y2006Q2);
27
[X_WoE2006Q3,InfoValue2006Q3,y2006Q3]  = GetAll20RiskDrivers('Acquisition_2006Q3.txt',y2006Q3);
28
[X_WoE2006Q4,InfoValue2006Q4,y2006Q4]  = GetAll20RiskDrivers('Acquisition_2006Q4.txt',y2006Q4);

[X_WoE2007Q1,InfoValue2007Q1,y2007Q1]  = GetAll20RiskDrivers('Acquisition_2007Q1.txt',y2007Q1);
30
[X_WoE2007Q2,InfoValue2007Q2,y2007Q2]  = GetAll20RiskDrivers('Acquisition_2007Q2.txt',y2007Q2);

[X_WoE2007Q3,InfoValue2007Q3,y2007Q3]  = GetAll20RiskDrivers('Acquisition_2007Q3.txt',y2007Q3);

[X_WoE2007Q4,InfoValue2007Q4,y2007Q4]  = GetAll20RiskDrivers('Acquisition_2007Q4.txt',y2007Q4);

[X_WoE2008Q1,InfoValue2008Q1,y2008Q1]  = GetAll20RiskDrivers('Acquisition_2008Q1.txt',y2008Q1);

[X_WoE2008Q2,InfoValue2008Q2,y2008Q2]  = GetAll20RiskDrivers('Acquisition_2008Q2.txt',y2008Q2);

[X_WoE2008Q3,InfoValue2008Q3,y2008Q3]  = GetAll20RiskDrivers('Acquisition_2008Q3.txt',y2008Q3);

[X_WoE2008Q4,InfoValue2008Q4,y2008Q4]  = GetAll20RiskDrivers('Acquisition_2008Q4.txt',y2008Q4);

[X_WoE2009Q1,InfoValue2009Q1,y2009Q1]  = GetAll20RiskDrivers('Acquisition_2009Q1.txt',y2009Q1);

[X_WoE2009Q2,InfoValue2009Q2,y2009Q2]  = GetAll20RiskDrivers('Acquisition_2009Q2.txt',y2009Q2);

[X_WoE2009Q3,InfoValue2009Q3,y2009Q3]  = GetAll20RiskDrivers('Acquisition_2009Q3.txt',y2009Q3);
40
[X_WoE2009Q4,InfoValue2009Q4,y2009Q4]  = GetAll20RiskDrivers('Acquisition_2009Q4.txt',y2009Q4);

[X_WoE2010Q1,InfoValue2010Q1,y2010Q1]  = GetAll20RiskDrivers('Acquisition_2010Q1.txt',y2010Q1);

[X_WoE2010Q2,InfoValue2010Q2,y2010Q2]  = GetAll20RiskDrivers('Acquisition_2010Q2.txt',y2010Q2);

[X_WoE2010Q3,InfoValue2010Q3,y2010Q3]  = GetAll20RiskDrivers('Acquisition_2010Q3.txt',y2010Q3);

[X_WoE2010Q4,InfoValue2010Q4,y2010Q4]  = GetAll20RiskDrivers('Acquisition_2010Q4.txt',y2010Q4);

[X_WoE2011Q1,InfoValue2011Q1,y2011Q1]  = GetAll20RiskDrivers('Acquisition_2011Q1.txt',y2011Q1);

[X_WoE2011Q2,InfoValue2011Q2,y2011Q2]  = GetAll20RiskDrivers('Acquisition_2011Q2.txt',y2011Q2);

[X_WoE2011Q3,InfoValue2011Q3,y2011Q3]  = GetAll20RiskDrivers('Acquisition_2011Q3.txt',y2011Q3);

[X_WoE2011Q4,InfoValue2011Q4,y2011Q4]  = GetAll20RiskDrivers('Acquisition_2011Q4.txt',y2011Q4);

[X_WoE2012Q1,InfoValue2012Q1,y2012Q1]  = GetAll20RiskDrivers('Acquisition_2012Q1.txt',y2012Q1);
50
[X_WoE2012Q2,InfoValue2012Q2,y2012Q2]  = GetAll20RiskDrivers('Acquisition_2012Q2.txt',y2012Q2);

[X_WoE2012Q3,InfoValue2012Q3,y2012Q3]  = GetAll20RiskDrivers('Acquisition_2012Q3.txt',y2012Q3);

[X_WoE2012Q4,InfoValue2012Q4,y2012Q4]  = GetAll20RiskDrivers('Acquisition_2012Q4.txt',y2012Q4);

[X_WoE2013Q1,InfoValue2013Q1,y2013Q1]  = GetAll20RiskDrivers('Acquisition_2013Q1.txt',y2013Q1);

[X_WoE2013Q2,InfoValue2013Q2,y2013Q2]  = GetAll20RiskDrivers('Acquisition_2013Q2.txt',y2013Q2);

[X_WoE2013Q3,InfoValue2013Q3,y2013Q3]  = GetAll20RiskDrivers('Acquisition_2013Q3.txt',y2013Q3);

[X_WoE2013Q4,InfoValue2013Q4,y2013Q4]  = GetAll20RiskDrivers('Acquisition_2013Q4.txt',y2013Q4);

[X_WoE2014Q1,InfoValue2014Q1,y2014Q1]  = GetAll20RiskDrivers('Acquisition_2014Q1.txt',y2014Q1);

[X_WoE2014Q2,InfoValue2014Q2,y2014Q2]  = GetAll20RiskDrivers('Acquisition_2014Q2.txt',y2014Q2);

[X_WoE2014Q3,InfoValue2014Q3,y2014Q3]  = GetAll20RiskDrivers('Acquisition_2014Q3.txt',y2014Q3);
60
[X_WoE2014Q4,InfoValue2014Q4,y2014Q4]  = GetAll20RiskDrivers('Acquisition_2014Q4.txt',y2014Q4);

[X_WoE2015Q1,InfoValue2015Q1,y2015Q1]  = GetAll20RiskDrivers('Acquisition_2015Q1.txt',y2015Q1);

[X_WoE2015Q2,InfoValue2015Q2,y2015Q2]  = GetAll20RiskDrivers('Acquisition_2015Q2.txt',y2015Q2);

[X_WoE2015Q3,InfoValue2015Q3,y2015Q3]  = GetAll20RiskDrivers('Acquisition_2015Q3.txt',y2015Q3);

[X_WoE2015Q4,InfoValue2015Q4,y2015Q4]  = GetAll20RiskDrivers('Acquisition_2015Q4.txt',y2015Q4);

[X_WoE2016Q1,InfoValue2016Q1,y2016Q1]  = GetAll20RiskDrivers('Acquisition_2016Q1.txt',y2016Q1);

[X_WoE2016Q2,InfoValue2016Q2,y2016Q2]  = GetAll20RiskDrivers('Acquisition_2016Q2.txt',y2016Q2);

[X_WoE2016Q3,InfoValue2016Q3,y2016Q3]  = GetAll20RiskDrivers('Acquisition_2016Q3.txt',y2016Q3);

[X_WoE2016Q4,InfoValue2016Q4,y2016Q4]  = GetAll20RiskDrivers('Acquisition_2016Q4.txt',y2016Q4);

[X_WoE2017Q1,InfoValue2017Q1,y2017Q1]  = GetAll20RiskDrivers('Acquisition_2017Q1.txt',y2017Q1);
70
[X_WoE2017Q2,InfoValue2017Q2,y2017Q2]  = GetAll20RiskDrivers('Acquisition_2017Q2.txt',y2017Q2);
71
[X_WoE2017Q3,InfoValue2017Q3,y2017Q3]  = GetAll20RiskDrivers('Acquisition_2017Q3.txt',y2017Q3);
72
[X_WoE2017Q4,InfoValue2017Q4,y2017Q4]  = GetAll20RiskDrivers('Acquisition_2017Q4.txt',y2017Q4);
73
[X_WoE2018Q1,InfoValue2018Q1,y2018Q1]  = GetAll20RiskDrivers('Acquisition_2018Q1.txt',y2018Q1);
73.5
[X_WoE2018Q2,InfoValue2018Q2,y2018Q2]  = GetAll20RiskDrivers('Acquisition_2018Q2.txt',y2018Q2);
74
[X_WoE2018Q3,InfoValue2018Q3,y2018Q3]  = GetAll20RiskDrivers('Acquisition_2018Q3.txt',y2018Q3);
75
[X_WoE2018Q4,InfoValue2018Q4,y2018Q4]  = GetAll20RiskDrivers('Acquisition_2018Q4.txt',y2018Q4);
