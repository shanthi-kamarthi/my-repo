library(NLP)
library(tm)
library(wordcloud)
library(e1071)#naive Bayes function
library(gmodels)
library("RColorBrewer")
vignette("tm")

sn_compliance=read.csv(file.choose(),header=T)
dim(sn_compliance)
y=factor(sn_compliance$Classification)
x_data=sn_compliance[,-17]
names(x_data)
head(y)
prop.table(table(y))

#Handling missing values
max(table(y))
for(i in 1:nrow(sn_compliance)){
  if(is.na(y[i])==TRUE)
  {
    y[i]="Preventive"   #max is Preventive
  }
}



prop.table(table(sn_compliance$Classification))
prop.table(table(y))
sn_compliance[1:5,]
attach(sn_compliance)


# combining all x columns in to on column
x_data = paste(Number,Name,Profile, Owner,Policy.Statement,State,Status,Exempt,Weighting,Active,Additional.Information,Additional.comments,      Attestation,Attestation.respondents,Category,Class,Created, Created.Manually,Created.by,Description,Domain,Domain.Path,Enforcement,Frequency,   From.configuration.check,Key.control,Owning.group,Profile.type,Profile.type.1,Source,Tags,Type,Updated,Updated.by,Updates,sep=" ")
x_data[1]

raw_data=data.frame(y,x_data)
dim(raw_data)
names(raw_data)=c("Classification","text")
raw_data$y=factor(sn_compliance$classification)
str(raw_data)


# create corpus
data_corpus=Corpus(VectorSource(raw_data$text))
inspect(data_corpus[1:3])
#4) Transformation or cleaning of corpus
corpus_clean=tm_map(data_corpus,tolower)#Cnvert everything into lwer case.
corpus_clean=tm_map(corpus_clean,removeNumbers)
corpus_clean=tm_map(corpus_clean,removeWords,stopwords())#Removing filler words are called Stopwords(to,but etc).
corpus_clean=tm_map(corpus_clean,removePunctuation)#Remove functuation
corpus_clean=tm_map(corpus_clean,stripWhitespace)
head(corpus_clean)

# Document term matrix
data_dtm=DocumentTermMatrix(corpus_clean)
inspect(data_dtm[1:2,1:3])


#sample generation
set.seed(1)
train=sample(nrow(data_dtm),nrow(data_dtm)/2)
test=(-train)

# splitting train and test data
y.test=raw_data$Classification[test]
y.test
y.train=raw_data$Classification[train]
#splitting of original data
data_train=raw_data[train,]
data_test=raw_data[test,]



#splitting of dtm
data_dtm_test=data_dtm[test,]
data_dtm_train=data_dtm[train,]

#splitting corpus_clean
data_corpus_train=corpus_clean[train]
data_corpus_test=corpus_clean[test]

#probability of preventive and detected in train and test data
prop.table(table(raw_data$Classification))
prop.table(table(data_train$Classification))
prop.table(table(data_test$Classification))

#6)Visualizing text data using wordcloud
wordcloud(data_corpus_train,min.freq=40,random.order=F,colors=brewer.pal(8, "Dark2"))

#word cloud for preventive
par(mfrow=c(1,2))
Preventive=data_train[data_train$Classification=="Preventive",]
wordcloud(Preventive$text,min.freq=40,random.order=F,colors=brewer.pal(8, "Dark2"))
#word cloud for Detective

Detective=data_train[data_train$Classification=="Detective",]
wordcloud(Detective$text,min.freq=40,random.order=F,colors=brewer.pal(8, "Dark2"))
par(mfrow=c(1,1))

#5) Creation of frequent data table
data_dict=(findFreqTerms(data_dtm_train,5))#Create the list of the words which seen more than 5 documents(1% of total documents)
data_text_train=DocumentTermMatrix(data_corpus_train,list(dictionary = data_dict))
data_text_test=DocumentTermMatrix(data_corpus_test,list(dictionary = data_dict))
data_text_train[1:2,c(2,3)]



#7) Creation of data table with updated with binary data in the table
#**************
#Create a function to convert the data "Yes" or "No"
convert_count=function(x){
  x=ifelse(x>0,1,0)
  x=factor(x,levels=c(0,1),labels=c("No","Yes"))
  return(x)
}
data_text_train=apply(data_text_train,MARGIN=2,convert_count)
data_text_test=apply(data_text_test,MARGIN=2,convert_count)


#8) Building the model
data_classifier=naiveBayes(data_text_train,y.train,laplace=1)


#9) Prediction using test data
data_test_pred=predict(data_classifier,newdata=data_text_test)
Performance=CrossTable(y.test,data_test_pred,prop.r=T,prop.t=F,prop.c=F,prop.chisq=F,dnn=c("Actual","Predicted"))


Accuracy=(Performance$t[1,2]+Performance$t[2,3])/length(y.test)
Accuracy
mean(y.test==data_test_pred)