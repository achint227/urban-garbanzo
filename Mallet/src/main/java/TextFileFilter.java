import java.io.File;
import java.io.FileFilter;

public class TextFileFilter implements FileFilter {
    @Override
    public boolean accept(File f){
        return f.toString().endsWith(".txt");
    }
}
