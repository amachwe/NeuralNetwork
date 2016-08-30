package rd.data;

import org.bson.Document;

public class WordMapMongoDecorator {

	public static enum Keys {
		Topic, Name, WordCount, DocCount, WordCountData, DocData, Type
	};



	public static Document getDocument(WordMap wordMap) {
		Document doc = new Document();
		Document wordCountDoc = new Document();
		wordCountDoc.putAll(wordMap.getWordCounts());
		
		doc.append(Keys.WordCountData.toString(), wordCountDoc);

		doc.append(Keys.DocData.toString(), wordMap.getDocIds());
		
		doc.append(Keys.Topic.toString(),wordMap.getTopic());
		
		doc.append(Keys.Name.toString(), wordMap.getName());
		
		doc.append(Keys.WordCount.toString(), wordMap.getWordCount());
		
		doc.append(Keys.DocCount.toString(), wordMap.getDocCount());
		
		doc.append(Keys.Type.toString(), wordMap.getType().toString());
		
		return doc;
	}
}
