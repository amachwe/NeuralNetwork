package rd.data;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;
import java.util.UUID;

public class FileDataWriter implements DataWriter {

	private final FileWriter fw;
	private final UUID batchId = UUID.randomUUID();
	private final boolean append;
	private final boolean fileExists;

	public FileDataWriter(String fileName, boolean append) throws IOException {
		fw = new FileWriter(fileName, append);
		this.append = append;
		fileExists = (new File(fileName)).length() > 0 ? true : false;
	}

	private boolean headerWritten = false;

	@Override
	public void write(Map<String, Object> row) {

		try {
			if (headerWritten) {
				StringBuilder rowData = new StringBuilder(batchId.toString());
				for (Object data : row.values()) {
					rowData.append(",");
					rowData.append(data);

				}
				rowData.append("\n");
				fw.write(rowData.toString());
			} else {
				StringBuilder rowData = new StringBuilder(batchId.toString());
				for (Object data : row.values()) {
					rowData.append(",");
					rowData.append(data);

				}
				rowData.append("\n");
				if (append && !fileExists) {
					String header = row.keySet().toString();
					fw.write("BatchId," + header.substring(1, header.length() - 1));
					fw.write("\n");
				}
				fw.write(rowData.toString());
				headerWritten = true;

			}
		} catch (IOException e) {
			System.err.println(e);
			e.printStackTrace();
			close();
		}
	}

	@Override
	public void close() {
		try {
			fw.close();
		} catch (IOException e) {
			System.err.println(e);
			e.printStackTrace();
		}

	}
}
