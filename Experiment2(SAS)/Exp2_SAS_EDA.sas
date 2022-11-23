data first;
	set sashelp.heart;
run;
title 5 point Summary;
proc means data=sashelp.heart mean median mode std var min max;
run;

Title NUmber of missing values;
proc means data=sashelp.heart nmiss;
run;

proc print =sashelp.heart;
where status = "Dead";
run;

title Getting Number of Distinct Values;
proc sql;
select count(distinct Status) as Status,
       count(distinct DeathCause) as DeathCause,
       count(distinct Sex) as Sex,
       count(distinct Chol_status) as Chol_status,
       count(distinct DeathCause) as Smoking_status
  from sashelp.heart;
quit;

title Correlation of the Attributes;
proc corr data=sashelp.heart;
run;

title Frequency of the Categorical Values
proc freq data=Sashelp.Heart; 
   tables _CHARACTER_;    /* _ALL_ is the defaul */
run;

proc print data= HeartNumeric(obs=5);
run;

proc means data=HeartNumeric nmiss;
run;

ods graphics / reset width=6.4in height=4.8in imagemap;
proc sgplot data=sashelp.heart;
	vbox  AgeAtStart / category=;
	yaxis grid;
run;
ods graphics / reset;

title "Scatter Plot of Height and weight";
proc sgplot data=sashelp.heart;
    scatter x = Height  y = Weight;
run;


title "Systolic Outlier";
proc sgplot data=sashelp.heart;
	vbox  Systolic / category=status;
	yaxis grid;
run;
ods graphics / reset;

title "Diastolic Outlier";
proc sgplot data=sashelp.heart;
	vbox  Diastolic / category=status;
	yaxis grid;
run;
ods graphics / reset;


title "Cholestrol ranges";
proc sgplot data=sashelp.heart;
	vbox  Cholesterol / category=Chol_Status;
	yaxis grid;
run;

DATA dead;
   SET sashelp.heart;
   IF (Status = "Dead") THEN OUTPUT;
RUN;
proc print =dead;
run;

title Diastolic BP Histogram grouped by BP_Status;
ods graphics / reset;
proc sort data=SASHELP.HEART out=_HistogramTaskData;
	by BP_Status;
run;
proc sgplot data=_HistogramTaskData;
	by BP_Status;
	histogram Diastolic /;
	yaxis grid;
run;
proc sgplot data=_HistogramTaskData;
	by BP_Status;
	histogram Systolic /;
	yaxis grid;
run;

