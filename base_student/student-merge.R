d1=read.table(file.choose(),sep=";",header=TRUE)
d2=read.table(file.choose(),sep=";",header=TRUE)

d3=merge(d1,d2,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))
print(nrow(d3)) # 382 students

d4 = rbind(d1, d2)
duplicated(d4)
d5 = unique(d4)
write.csv(d4, file="students_all.csv")

a = c(1, 2, 3, 5, 5, 5)
b = duplicated(a)