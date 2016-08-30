package rd.data.test;

import java.io.File;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import org.apache.commons.io.FileUtils;
import org.bson.Document;
import org.junit.Test;

import com.aliasi.tokenizer.EnglishStopTokenizerFactory;
import com.aliasi.tokenizer.IndoEuropeanTokenizerFactory;
import com.aliasi.tokenizer.Tokenizer;
import com.aliasi.tokenizer.TokenizerFactory;
import com.mongodb.MongoClient;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.model.InsertManyOptions;

import rd.data.DirectoryParser;
import rd.data.WordMap;
import rd.data.WordMap.Type;
import rd.data.WordMapImpl;
import rd.data.WordMapMongoDecorator;

public class TestDirectoryParser {
	private static final String DB_NAME = "NLP";

	private final String BASE = "data\\bbc-fulltext\\bbc";

	private final String URL = "192.168.0.17";
	private final long SIZE = 2225;

	private static final TokenizerFactory stopWordTokenizer = new EnglishStopTokenizerFactory(
			IndoEuropeanTokenizerFactory.INSTANCE);

	private final MongoClient mc = new MongoClient(URL, 27017);

	private final InsertManyOptions options = new InsertManyOptions();
	{
		options.ordered(false);
	}

	@Test
	public void doTest() throws InterruptedException, ExecutionException {
		Set<Long> step1=new HashSet<>();
		Set<Long> step2=new HashSet<>();
		for(int i = 0;i<4;i++)
		{
		(new TestDirectoryParser()).process(step1,step2);
		}
		
		System.out.println(step1+"\n\n"+step2);
		

	}

	public void process(Set<Long> step1,Set<Long> step2) throws InterruptedException, ExecutionException {
		ForkJoinPool pool = new ForkJoinPool(4);

		mc.dropDatabase(DB_NAME);
		MongoCollection<Document> docColl = mc.getDatabase(DB_NAME).getCollection("documents");
		MongoCollection<Document> topicColl = mc.getDatabase(DB_NAME).getCollection("topics");
		WordMap CORPUS = new WordMapImpl("CORPUS", "CORPUS", Type.Corpus);
		ConcurrentLinkedQueue<Document> docsList = new ConcurrentLinkedQueue<>();
		ConcurrentLinkedQueue<Document> topicsList = new ConcurrentLinkedQueue<>();
		Set<File> files = DirectoryParser.getFiles(new File(BASE), DirectoryParser.TEXT_FILE_FILTER);
		long startTime = System.currentTimeMillis();
		pool.submit(() -> {
			try {
				
				System.out.println("--- Streaming ---");
				Stream<File> fileStream = StreamSupport.stream(files.spliterator(), true);
				fileStream.parallel().map(file -> {

					String topicName = file.getParent();
					String name = file.getName();
					char[] text = new char[0];
					try {
						text = FileUtils.readFileToString(file).toCharArray();
					} catch (IOException e) {
						System.err.println(e);
						e.printStackTrace();
					}
					Tokenizer tokenizer = stopWordTokenizer.tokenizer(text, 0, text.length);
					final WordMap doc = new WordMapImpl(topicName, name, WordMap.Type.Document);
					final WordMap topic = new WordMapImpl(topicName, topicName, WordMap.Type.Topic);

					tokenizer.forEach(str -> {
						if (check(str)) {
							doc.add(str, name);
							topic.add(str, name);
							CORPUS.addOncePerDoc(str, topicName + "_" + name);

						}
					});
					if (doc != null) {
						docsList.add(WordMapMongoDecorator.getDocument(doc));
					}

					return topic;

				}).collect(Collectors.groupingBy(WordMap::getTopic))
						.forEach((name, topics) -> {
							WordMap topicName = new WordMapImpl(name, name, WordMap.Type.Topic);
							topics.stream().forEach(topic -> {
								topicName.accumulate(topic);

							});

							topicsList.add(WordMapMongoDecorator.getDocument(topicName));
						});
				
			
			} catch (Exception e) {
				e.printStackTrace();
				System.out.println(e);
			}

		}).get();
		step1.add(System.currentTimeMillis() - startTime);
		startTime = System.currentTimeMillis();
		System.out.println("--- Writing ---");
		assert(docsList.size()== SIZE);
		docColl.insertMany(docsList.stream().parallel().collect(Collectors.toList()),options);
		topicColl.insertMany(topicsList.stream().parallel().collect(Collectors.toList()),options);
		topicColl.insertOne(WordMapMongoDecorator.getDocument(CORPUS));
		step2.add(System.currentTimeMillis() - startTime);
		assert (files.size() == SIZE);

	}

	private static final boolean check(String str) {
		String _str = str.trim();
		if (_str.length() > 1) {
			if (!_str.contains(".")) {
				return true;
			}
		}

		return false;
	}
}
