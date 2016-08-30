package rd.data.test;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

import org.bson.Document;

import com.google.common.io.Files;

import rd.data.CurrencyClient;
import rd.data.CurrencyClientImpl;
import rd.data.MovementData;
import rd.data.MovementData.Keys;

public class TestCurrencyClient {

	private static final String[] currCodes = { "GBP", "INR", "USD", "NGN", "CNY", "EUR" },
			targetCodes = { "XDR", "XAU", "USD" };
	private static final String base = "rates", TS = "timestamp";

	public final static void main(String[] args) throws InterruptedException, ExecutionException {

		List<MovementData> invalidRecords = new ArrayList<>();
		CurrencyClient cc = new CurrencyClientImpl("http://localhost:4567/currency/");
		String postFix = ".csv";
		String dataFile = "data/currency/result/result_full" + postFix;
		String errorFile = "data/currency/result/error/error_full" + postFix;

		Path resultPath = Paths.get(dataFile);
		Path errorPath = Paths.get(errorFile);
		try (BufferedWriter w = Files.newWriter(resultPath.toAbsolutePath().toFile(), Charset.defaultCharset());
				BufferedWriter e = Files.newWriter(errorPath.toAbsolutePath().toFile(), Charset.defaultCharset())) {
			for (int i = 0; i < currCodes.length; i++) {
				for (int j = i + 1; j < currCodes.length; j++) {
					for (int t = 0; t < targetCodes.length; t++) {

						String currA = currCodes[i], currB = currCodes[j], target = targetCodes[t];
						if (!target.equalsIgnoreCase(currA) && !target.equalsIgnoreCase(currB)) {
							MovementData.setCurrencyKeys(currA, currB, target);
							List<MovementData> pairs = new ArrayList<>();
							List<Document> docList = cc.getCurrencyPair(currA, currB, target);
							StreamSupport.stream(docList.spliterator(), false).reduce(null, (_prev, _curr) -> {

								if (_prev != null && _curr != null) {
									Map<Keys, Number> data = new HashMap<>();
									Document currRates = ((Document) _curr.get(base));

									Document prevRates = ((Document) _prev.get(base));
									data.put(Keys.CurrentA, getValue(currRates.get(currA)));
									data.put(Keys.CurrentB, getValue(currRates.get(currB)));
									data.put(Keys.CurrentTarget, getValue(currRates.get(target)));

									data.put(Keys.PreviousA, getValue(prevRates.get(currA)));
									data.put(Keys.PreviousB, getValue(prevRates.get(currB)));
									data.put(Keys.PreviousTarget, getValue(prevRates.get(target)));
									MovementData md = MovementData.build(_curr.getInteger(TS), _prev.getInteger(TS),
											data);
									if (md.isValid() && md.getRange() < 4000) {
										pairs.add(md);
									} else {
										if (!md.isValid()) {
											invalidRecords.add(md);
										}
									}

								}
								return _curr;
							});

							ForkJoinPool pool = new ForkJoinPool(4);
							Map<String, Long> map = pool.submit(new ForkJoinTask<Map<String, Long>>() {
								/**
								 * 
								 */
								private static final long serialVersionUID = -5199890223313593220L;
								private Map<String, Long> map = null;

								@Override
								protected boolean exec() {
									setRawResult(pairs.stream().parallel().collect(Collectors
											.groupingByConcurrent(MovementData::getTrendKey, Collectors.counting())));

									return true;
								}

								@Override
								public Map<String, Long> getRawResult() {

									return map;
								}

								@Override
								protected void setRawResult(Map<String, Long> arg0) {
									map = arg0;

								}
							}).get();

							long sum = 0;
							for (String k : map.keySet()) {
								sum += map.get(k);
							}
							process(map, w, sum);

							System.out.println(currA + " - " + currB + " - " + target + " - " + sum);

							for (MovementData d : invalidRecords) {
								e.append(d.toString());
								e.newLine();
							}

						}
					}
				}
			}
		} catch (IOException ioe) {
			ioe.printStackTrace();
		} finally {
			System.out.println(totals);
		}

	}

	private static Map<String, Long> totals = new HashMap<>();

	private static void process(Map<String, Long> map, BufferedWriter w, long total) {
		map.forEach((k, v) -> {
			try {
				String data[] = k.split(",");

				if (data.length == 6) {
					String reduced_key = data[4] + data[5];
					String total_key = "total"+data[5];
					Long temp = 0L;
					totals.put(total_key, totals.getOrDefault(total_key, 0L) + v);
					if ((temp = totals.get(reduced_key)) != null) {
						totals.put(reduced_key, temp + v);

					} else {
						totals.put(reduced_key, v);

					}
				}
				w.append(k + ", " + v + "," + (v * 100d / total));
				w.newLine();
			} catch (IOException ioe) {
				ioe.printStackTrace();
			}
		});
	}

	private static final Double getValue(Object val) {
		if (val == null) {
			return -1d;
		}
		if (val instanceof Integer) {
			return 1d * (Integer) val;
		} else {
			return (Double) val;
		}
	}
}
