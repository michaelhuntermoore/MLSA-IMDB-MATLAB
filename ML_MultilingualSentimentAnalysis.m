%Hunter Moore
%Machine Learning Semester Project: Multilingual Sentiment Analysis
clc;
clear;
%Read English samples (30 positive, 30 negative IMDb reviews)
file_samplesENG = fopen('SamplesENG.txt');
C = textscan(file_samplesENG,'%s','Delimiter','\n');
samplesENG = string(C{1});
%Read Machine-translated Spanish samples
file_samplesESP = fopen('SamplesENGtoESP.txt');
C = textscan(file_samplesESP,'%s','Delimiter','\n');
samplesESP = string(C{1});
%Read ENG->ESP->ENG samples
file_samples_final = fopen('SamplesENGtoESPtoENG.txt');
C = textscan(file_samples_final,'%s','Delimiter','\n');
samples_final = string(C{1});

%Wordcloud visualization 
 %Wordclouds without preprocessing
   figure(1);
  subplot(1,2,1);
   wordcloud(samplesENG);
     title("Wordcloud for English Samples before preprocessing");
   figure(2);
  subplot(1,2,1);
   wordcloud(samplesESP);
     title("Wordcloud for ENG->ESP Samples before preprocessing");
     figure(3);
  subplot(1,2,1);
   wordcloud(samples_final);
     title("Wordcloud for ENG->ESP->ENG Samples before preprocessing");
%--------------------------------------
%Preprocessing

%Change all text data to lowercase
cleanTextData1 = lower(samplesENG);
cleanTextData2 = lower(samplesESP);
cleanTextData3 = lower(samples_final);
cleanTextData1(1:10)

%Tokenize data
cleanDocuments1 = tokenizedDocument(cleanTextData1);
cleanDocuments2 = tokenizedDocument(cleanTextData2);
cleanDocuments3 = tokenizedDocument(cleanTextData3);
cleanDocuments1(1:10)
%Remove punctuation
cleanDocuments1 = erasePunctuation(cleanDocuments1);
cleanDocuments2 = erasePunctuation(cleanDocuments2);
cleanDocuments3 = erasePunctuation(cleanDocuments3);
%Remove stop words
cleanDocuments1 = removeStopWords(cleanDocuments1);
cleanDocuments2 = removeStopWords(cleanDocuments2);
cleanDocuments3 = removeStopWords(cleanDocuments3);
%Remove words that are too short or too long
cleanDocuments1 = removeShortWords(cleanDocuments1,2);
cleanDocuments1 = removeLongWords(cleanDocuments1,25);
cleanDocuments2 = removeShortWords(cleanDocuments2,2);
cleanDocuments2 = removeLongWords(cleanDocuments2,25);
cleanDocuments3 = removeShortWords(cleanDocuments3,2);
cleanDocuments3 = removeLongWords(cleanDocuments3,25);
cleanDocuments1(1:10)
%Lemmatize the words
cleanDocuments1 = addPartOfSpeechDetails(cleanDocuments1);
cleanDocuments1 = normalizeWords(cleanDocuments1,'Style','lemma');
cleanDocuments2 = addPartOfSpeechDetails(cleanDocuments2);
cleanDocuments2 = normalizeWords(cleanDocuments2,'Style','lemma');
cleanDocuments3 = addPartOfSpeechDetails(cleanDocuments3);
cleanDocuments3 = normalizeWords(cleanDocuments3,'Style','lemma');
cleanDocuments1(1:10);
%Remove infrequent words
bag1 = bagOfWords(cleanDocuments1);
cleanBag1 = removeInfrequentWords(bag1,2);
bag2 = bagOfWords(cleanDocuments2);
cleanBag2 = removeInfrequentWords(bag2,2);
bag3 = bagOfWords(cleanDocuments3);
cleanBag3 = removeInfrequentWords(bag3,2);
%Remove empty documents
cleanBag1 = removeEmptyDocuments(cleanBag1);
cleanBag2 = removeEmptyDocuments(cleanBag2);
cleanBag3 = removeEmptyDocuments(cleanBag3); 

figure(4); 
subplot(1,2,2);
wordcloud(cleanBag1);
title("Wordcloud for English Samples after preprocessing");
figure(5); 
subplot(1,2,2);
wordcloud(cleanBag2);
title("Wordcloud for ENG->ESP Samples after preprocessing");
figure(6); 
subplot(1,2,2);
wordcloud(cleanBag3);
title("Wordcloud for ENG->ESP->ENG Samples after preprocessing");
