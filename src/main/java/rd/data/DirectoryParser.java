package rd.data;

import java.io.File;
import java.io.FileFilter;
import java.util.List;

public class DirectoryParser {

	public static void getFiles(List<File> allFiles,File rootDir, final FileFilter fileFilter) {
	

		File[] fileList = rootDir.listFiles(fileFilter);
		if (fileList != null && fileList.length > 0) {
			for (File file : fileList) {
				if (file.isDirectory()) {
					getFiles(allFiles,file, fileFilter);
				} else {
					allFiles.add(file);
				}
			}
		}
		
	}
	
	public static void getFiles(List<File> allFiles,File rootDir, final FileFilter fileFilter, boolean verbose) {
		

		if(verbose)
		{
			System.out.println("Processing: "+rootDir);
		}
		File[] fileList = rootDir.listFiles(fileFilter);
		if(verbose)
		{
			System.out.println("Got file list for: "+rootDir);
		}
		if (fileList != null && fileList.length > 0) {
			for (File file : fileList) {
				if(verbose)
				{
					System.out.println("Check: "+ file);
				}
				if (file.isDirectory()) {
					if(verbose)
					{
						System.out.print(" -  Directory "+allFiles.size());
					}
					getFiles(allFiles,file, fileFilter,verbose);
					
				} else {
					if(verbose)
					{
						System.out.println("Added: "+ file);
					}
					allFiles.add(file);
				}
			}
		}
		
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
	
	public static final FileFilter IMG_FILE_FILTER = new FileFilter() {

		@Override
		public boolean accept(File arg0) {
			if(arg0==null)
			{
				return false;
			}
			if (arg0.isDirectory() || (arg0.isFile() && (arg0.getName().endsWith(".png")|| arg0.getName().endsWith(".jpg") || arg0.getName().endsWith(".gif")))) {
				return true;
			}
			return false;
		}
	};
	
	

}
