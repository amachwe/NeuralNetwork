package rd.data;

import java.util.List;

import org.bson.Document;

public interface CurrencyClient {

	List<Document> getCurrencyPair(String currA, String currB, String target);

}