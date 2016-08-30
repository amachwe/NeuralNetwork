package rd.data;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class WordMapImpl implements WordMap {
	private final Map<String, Integer> wordCounts = new HashMap<>();
	private final Set<String> docIds = new HashSet<>();
	private final String topic;
	private final String name;

	private final Type type;

	/**
	 * For a document word map
	 * 
	 * @param topic
	 * @param name
	 */
	public WordMapImpl(String topic, String name, Type type) {
		this.topic = topic;
		this.name = name;
		this.type = type;

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see rd.data.WordMap#add(java.util.List)
	 */
	@Override
	public void add(List<String> words, String docId) {

		words.forEach(word -> {
			add(word, docId);
		});
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see rd.data.WordMap#add(java.lang.String)
	 */
	@Override
	public synchronized void add(String word, String docId) {
		docIds.add(docId);

		Integer val = null;
		if ((val = wordCounts.get(word)) != null) {
			wordCounts.put(word, val + 1);
		} else {
			wordCounts.put(word, 1);
		}

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see rd.data.WordMap#getTopic()
	 */
	@Override
	public String getTopic() {
		return topic;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see rd.data.WordMap#getName()
	 */
	@Override
	public String getName() {
		return name;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see rd.data.WordMap#getWordCount()
	 */
	@Override
	public int getWordCount() {
		return wordCounts.size();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see rd.data.WordMap#toString()
	 */
	@Override
	public String toString() {
		StringBuilder strBuilder = new StringBuilder("Topic: ");
		strBuilder.append(topic);
		strBuilder.append("\tName: ");
		strBuilder.append(name);
		strBuilder.append("\tCounts: ");
		strBuilder.append(this.wordCounts.size());
		strBuilder.append("\tDocs: ");
		strBuilder.append(this.docIds.size());
		strBuilder.append("\n");
		strBuilder.append(wordCounts);
		strBuilder.append("\n");
		return strBuilder.toString();
	}

	@Override
	public int getDocCount() {
		return docIds.size();
	}

	@Override
	public Set<String> getDocIds() {
		return docIds;
	}

	@Override
	public Type getType() {
		return type;
	}

	@Override
	public Map<String, Integer> getWordCounts() {
		return this.wordCounts;
	}

	@Override
	public synchronized void accumulate(WordMap map) {
		if (this.topic.equalsIgnoreCase(map.getTopic()) && this.name.equalsIgnoreCase(map.getName())) {
			Integer count = null;
			for (String word : map.getWordCounts().keySet()) {

				if ((count = wordCounts.get(word)) == null) {
					this.wordCounts.put(word, map.getWordCounts().get(word));
				} else {

					this.wordCounts.put(word, count + map.getWordCounts().get(word));
				}
			}

			this.docIds.addAll(map.getDocIds());

		} else {
			System.err.println("Error: Topic and Name do not match");

		}
	}

	@Override
	public void addOncePerDoc(List<String> words, String docId) {
		words.forEach(word -> {
			addOncePerDoc(word, docId);
		});

	}

	@Override
	public synchronized void addOncePerDoc(String word, String docId) {

		if (!(docIds.contains(docId) && this.wordCounts.containsKey(word))) {
			docIds.add(docId);

			Integer val = null;
			if ((val = wordCounts.get(word)) != null) {
				this.wordCounts.put(word, val + 1);
			} else {
				this.wordCounts.put(word, 1);
			}
		}

	}

}
