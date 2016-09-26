package rd.data;

import java.io.File;
import java.io.PrintWriter;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Maintain distribution (count) data between two variables V and K, can be used
 * for calculating joint probabilities
 * 
 * @author azahar
 *
 * @param <V>
 * @param <K>
 */
public class DistributionStructure<V, K> {

	private static final Logger logger = LoggerFactory.getLogger(DistributionStructure.class);
	private ConcurrentHashMap<V, Integer> rowIndex = new ConcurrentHashMap<>();
	private ConcurrentHashMap<K, Integer> colIndex = new ConcurrentHashMap<>();

	private final AtomicInteger currRow = new AtomicInteger(0), currCol = new AtomicInteger(0);
	private final AtomicInteger totalCount = new AtomicInteger(0);
	private final int[][] count;
	private final int maxRow, maxCol;
	private String rowLabel, colLabel;

	/**
	 * 
	 * @param _maxRow
	 *            - maximum number of rows
	 * @param _maxCol
	 *            - maximum number of cols
	 */
	public DistributionStructure(int _maxRow, int _maxCol) {
		count = new int[_maxRow][_maxCol];
		maxRow = _maxRow;
		maxCol = _maxCol;
	}

	/**
	 * Set row and column labels
	 * @param rowLabel
	 * @param colLabel
	 */
	public void addLabels(String rowLabel, String colLabel) {
		this.rowLabel = rowLabel;
		this.colLabel = colLabel;
	}

	public String getRowLabel() {
		return this.rowLabel;
	}

	public String getColumnLabel() {
		return this.colLabel;
	}

	public void add(V rowKey, K colKey) throws Exception {
		Integer rowI = rowIndex.get(rowKey);
		if (rowI == null) {
			rowI = currRow.getAndIncrement();
			rowIndex.put(rowKey, rowI);
		}

		Integer colI = colIndex.get(colKey);
		if (colI == null) {
			colI = currCol.getAndIncrement();
			colIndex.put(colKey, colI);
		}

		if (maxRow > rowI && maxCol > colI) {
			count[rowI][colI] += 1;
			totalCount.incrementAndGet();

		} else {
			throw new Exception("Space overrun, row or column index exceeds set bounds. Row Index: " + rowI
					+ "   Col Index: " + colI + ";  Max Row/Col: " + maxRow + " / " + maxCol);
		}

	}

	public Integer getCurrentRowCount() {
		return currRow.get();
	}

	public Integer getCurrentColumnCount() {
		return currCol.get();
	}

	/**
	 * 
	 * @param rowKey
	 * @param colKey
	 * @return A valid value >= 0 ; -1 on error
	 */
	public Integer get(V rowKey, K colKey) {
		Integer rowI = rowIndex.get(rowKey);
		Integer colI = colIndex.get(colKey);
		if (colI == null || rowI == null) {
			logger.error("Invalid index for row or column key: " + rowKey + "   " + colKey);
			return -1;
		}
		if (colI >= maxCol || rowI >= maxRow) {
			logger.error("Index larger than size for row or column: " + rowKey + "   " + colKey);
			return -1;
		}

		return count[rowI][colI];

	}

	public void writeToFile(File file, boolean normalise) {

		try (PrintWriter pw = new PrintWriter(file)) {
			print(pw, normalise);
		} catch (Exception e) {
			logger.error(e.getMessage(), e);

		}

	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder("\n\t,");
		for (K colKey : colIndex.keySet()) {
			sb.append(colKey);
			sb.append(",\t");
		}
		sb.append("\n");

		for (V rowKey : rowIndex.keySet()) {
			Integer rowI = rowIndex.get(rowKey);
			sb.append(rowKey);
			sb.append(",\t");
			for (int i = 0; i < currCol.get(); i++) {
				sb.append(count[rowI][i]);
				sb.append(",\t");
			}
			sb.append("\n");
		}
		sb.append("\n");
		return sb.toString();
	}

	public void print(PrintWriter w, boolean normalise) {
		float sum = totalCount.get();
		if (sum <= 0) {
			sum = 1;
		}
		w.print("RowId");
		for (K colKey : colIndex.keySet()) {
			w.print(",");
			w.print(colKey);

		}
		w.println();

		for (V rowKey : rowIndex.keySet()) {

			Integer rowI = rowIndex.get(rowKey);
			w.print(rowKey);

			for (int i = 0; i < currCol.get(); i++) {

				w.print(",");

				if (normalise) {
					w.print(count[rowI][i] / sum);
				} else {
					w.print(count[rowI][i]);
				}

			}
			w.println();
		}
		w.println();

	}
}
