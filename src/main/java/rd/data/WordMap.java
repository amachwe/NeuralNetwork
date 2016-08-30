package rd.data;

import java.util.List;
import java.util.Map;
import java.util.Set;

public interface WordMap {

	public enum Type {Document,Topic,Corpus};
	
	void add(List<String> words,String docId);
	
	void addOncePerDoc(List<String> words,String docId);

	void addOncePerDoc(String word,String docId);
	void add(String word,String docId);

	String getTopic();

	String getName();

	int getWordCount();
	
	int getDocCount();

	String toString();
	
	Set<String> getDocIds();
	
	Type getType();
	
	Map<String,Integer> getWordCounts();
	
	void accumulate(WordMap map);

}