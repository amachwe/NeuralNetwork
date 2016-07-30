package rd.data;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.bson.Document;

import com.mongodb.MongoClient;
import com.mongodb.client.MongoCollection;

public class MongoDeltaWriter implements DataWriter {

	private final List<Document> buffer = new ArrayList<>();
	private final int MAX_BUFFER_SIZE = 1000;
	private final MongoClient client;
	private final MongoCollection<Document> collection;

	/**
	 * 
	 * @param host
	 * @param port
	 * @param dbName
	 * @param collectionName
	 */
	public MongoDeltaWriter(String host, int port, String dbName, String collectionName) {

		this.client = new MongoClient(host, port);

	
		this.collection = client.getDatabase(dbName).getCollection(collectionName);

	}

	private void processBuffer(List<Document> buffer) {

		collection.insertMany(buffer);
		System.out.println("Batch written");
		buffer.clear();

	}

	@Override
	public void write(Map<String, Object> data) {
		Document doc = new Document(data);
		insert(doc);

	}

	public void insert(Document doc) {

		buffer.add(doc);

		if (buffer.size() >= MAX_BUFFER_SIZE) {
			processBuffer(buffer);
		}

	}

	public void close() {

		if (buffer.size() > 0)
			collection.insertMany(buffer);

		client.close();

	}

}
