package rd.data;

import java.io.File;
import java.io.FileFilter;
import java.util.HashSet;
import java.util.Set;

public class DirectoryParser {

	public static Set<File> getFiles(File rootDir, final FileFilter textFileFilter) {
		Set<File> files = new HashSet<>();

		File[] fileList = rootDir.listFiles(textFileFilter);
		if (fileList != null && fileList.length > 0) {
			for (File file : fileList) {
				if (file.isDirectory()) {
					files.addAll(getFiles(file, textFileFilter));
				} else {
					files.add(file);
				}
			}
		}
		return files;
	}

	public static final FileFilter TEXT_FILE_FILTER = new FileFilter() {

		@Override
		public boolean accept(File arg0) {
			if(arg0==null)
			{
				return false;
			}
			if (arg0.isDirectory() || (arg0.isFile() && arg0.getName().endsWith(".txt"))) {
				return true;
			}
			return false;
		}
	};

}
