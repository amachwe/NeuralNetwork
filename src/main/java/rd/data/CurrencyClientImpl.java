package rd.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.zip.GZIPInputStream;

import org.bson.Document;

public class CurrencyClientImpl implements CurrencyClient {

	private final String GET = "GET";
	private final String rooturl;
	private final List<Document> EMPTY = Collections.unmodifiableList(new ArrayList<>());

	/**
	 * Root url for requests http://host:port/currency/
	 * 
	 * @param rooturl
	 */
	public CurrencyClientImpl(String rooturl) {
		this.rooturl = rooturl;
	}

	private List<Document> sendGzipPairRequest(String currA, String currB, String target) throws IOException {
		List<Document> result = new ArrayList<Document>();
		URL url = new URL(rooturl + currA + "/" + currB + "/" + target + "?gzip");
		HttpURLConnection conn = (HttpURLConnection) url.openConnection();
	
		conn.setRequestMethod(GET);
		int responseCode = conn.getResponseCode();

		BufferedReader br = new BufferedReader(new InputStreamReader(new GZIPInputStream(conn.getInputStream())));
		String line = "";
		Document meta = null;
		while ((line = br.readLine()) != null) {
			Document doc = Document.parse(line);
			if(!doc.containsKey("meta_count"))
			{
				result.add(doc);
			}
			else
			{
				meta = doc;
			}
		}
		br.close();
		
		if(meta==null)
		{
			throw new IllegalStateException("Bad metadata, request may not have ended properly.");
		}
		int expectedDocCounts = meta.getInteger("meta_count", -1);
		if(expectedDocCounts!=result.size())
		{
			throw new IllegalStateException("BSON Object Counts do not match the expected result size. Expected: "+expectedDocCounts+"  Found: "+ result.size());
		}
		return result;

	}

	/* (non-Javadoc)
	 * @see rd.data.CurrencyClient#getCurrencyPair(java.lang.String, java.lang.String, java.lang.String)
	 */
	@Override
	public List<Document> getCurrencyPair(String currA, String currB, String target) {

		try {
			return sendGzipPairRequest(currA, currB, target);
		} catch (IOException e) {
		
			e.printStackTrace();
			return EMPTY;
		}
	}

}
