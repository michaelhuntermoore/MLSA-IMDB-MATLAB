%Hunter Moore
clc;
%load pretrained word embedding
emb = fastTextWordEmbedding;
 
%Load Opinion Lexicon
data = readLexicon;
 
%Train the Sentiment Classifier
idx = ~isVocabularyWord(emb,data.Word);
data(idx,:) = [];
numWords = size(data,1);
cvp = cvpartition(numWords,'HoldOut',0.1);
dataTrain = data(training(cvp),:);
dataTest = data(test(cvp),:);
wordsTrain = dataTrain.Word;
XTrain = word2vec(emb,wordsTrain);
YTrain = dataTrain.Label;

%Fit model to an SVM 
mdl2 = fitcsvm(XTrain,YTrain);
wordsTest = dataTest.Word;
XTest = word2vec(emb,wordsTest);
YTest = dataTest.Label;
[YPred,scores] = predict(mdl2,XTest);

%Plot confusion matrix
figure
confusionchart(YTest,YPred);
 
%Wordcloud visualization of trained pos./neg. sentiment
figure
 subplot(1,2,1)
 idx = YPred == "Positive";
 wordcloud(wordsTest(idx),scores(idx,1));
 title("Predicted Positive Sentiment")
 subplot(1,2,2)
 wordcloud(wordsTest(~idx),scores(~idx,2));
 title("Predicted Negative Sentiment")
 
 
 
 %Calculate Sentiment for a given sample and/or set of samples
 textData1 = samplesENG;
 textData2 = samplesESP;
 textData3 = samples_final;
 
documents1 = preprocessReviews(textData1);
idx = ~isVocabularyWord(emb,documents1.Vocabulary);
documents1 = removeWords(documents1,idx);
 
idx = 1:60;
for i = 1:numel(idx)
    words = string(documents1(idx(i)));
    vec = word2vec(emb,words);
    [~,scores] = predict(mdl2,vec);
    sentimentScore1(i) = mean(scores(:,1));
    
end
[sentimentScore1' textData1(idx)]

documents2 = preprocessReviews(textData2);
idx = ~isVocabularyWord(emb,documents2.Vocabulary);
documents2 = removeWords(documents2,idx);
 
idx = 1:60;
for i = 1:numel(idx)
    words = string(documents2(idx(i)));
    vec = word2vec(emb,words);
    [~,scores] = predict(mdl2,vec);
    sentimentScore2(i) = mean(scores(:,1));
    
end
[sentimentScore2' textData2(idx)]

documents3 = preprocessReviews(textData3);
idx = ~isVocabularyWord(emb,documents3.Vocabulary);
documents3 = removeWords(documents3,idx);
idx = 1:60;
for i = 1:numel(idx)
    words = string(documents3(idx(i)));
    vec = word2vec(emb,words);
    [~,scores] = predict(mdl2,vec);
    sentimentScore3(i) = mean(scores(:,1));
    
end
[sentimentScore3' textData3(idx)]



 %This function only needs to run once in the MATLAB workspace -- for importing the opinion lexicon.
function data = readLexicon

% Read positive words
fidPositive = fopen(fullfile('opinion-lexicon-English','positive-words.txt'));
C = textscan(fidPositive,'%s','CommentStyle',';');
wordsPositive = string(C{1});

% Read negative words
fidNegative = fopen(fullfile('opinion-lexicon-English','negative-words.txt'));
C = textscan(fidNegative,'%s','CommentStyle',';');
wordsNegative = string(C{1});
fclose all;

% Create table of labeled words
words = [wordsPositive;wordsNegative];
labels2 = categorical(nan(numel(words),1));
labels2(1:numel(wordsPositive)) = "Positive";
labels2(numel(wordsPositive)+1:end) = "Negative";

data = table(words,labels2,'VariableNames',{'Word','Label'});

end
 
function [documents] = preprocessReviews(textData)
 
% Convert the text data to lowercase.
cleanTextData = lower(textData);
 
% Tokenize the text.
documents = tokenizedDocument(cleanTextData);
 
% Erase punctuation.
documents = erasePunctuation(documents);
 
% Remove a list of stop words.
documents = removeStopWords(documents);
%Remove long and short words
documents = removeShortWords(documents,2);
documents = removeLongWords(documents,25);
%Stem and lemmatize the documents
documents = addPartOfSpeechDetails(documents);
documents = normalizeWords(documents,'Style','lemma');
end