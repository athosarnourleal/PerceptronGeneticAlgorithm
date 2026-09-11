import java.awt.Canvas;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Rectangle;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.image.BufferStrategy;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import javax.swing.JFrame;

public class Visualizer extends Canvas implements Runnable, KeyListener {

	private static final long serialVersionUID = 1L;
	public String title = "score improvement viewer";
	public static int T = 16, W = 100*T,H = 60*T;

	public static int totalW;

	public boolean isRunning = false;

	public BufferedImage screen = new BufferedImage(W,H,BufferedImage.TYPE_4BYTE_ABGR_PRE);
	public JFrame frame;
	public Thread thread = new Thread(this);
	public int frameRate = 0;

	public ArrayList<Double> dataSetScore = new ArrayList<>();

	BufferedReader fileReader;
	public int collumnW = 10;

	public int cameraX = 0;
	public int maxVx = 25, vx = 0;

	public Visualizer() {
		this.setPreferredSize(new Dimension(W,H));
		this.addKeyListener(this);

		try {
			fileReader = new BufferedReader(new FileReader("scores.txt"));
			extractDataScore();
		} catch(IOException e) {
		  e.printStackTrace();
		}

		totalW = dataSetScore.size()*collumnW;

		System.out.println("data extracted.");
		System.out.println("loaded: "+dataSetScore.size());

		initFrame();
		thread.start();
	}

	public void initFrame() {
		frame = new JFrame(title);

		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.add(this);
		frame.pack();
		frame.setVisible(true);
		frame.setResizable(true);
		frame.setLocationRelativeTo(null);
		requestFocus();
	}



	public void extractDataScore() throws IOException {
		dataSetScore = new ArrayList<Double>();

		String line;
		while ((line = fileReader.readLine()) != null) {
			dataSetScore.add(Double.parseDouble(line));
		}
	}

    // copied from Arduino
	public double map(double x, double in_min, double in_max, double out_min, double out_max) {
		return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
	}

	public void drawLines(ArrayList<Double> dataset,  Graphics g) {

		// finding max value
		double maxVal = dataset.get(0);
		double minVal = dataset.get(0);

		for (int i = 1; i < dataset.size(); i++) {
			if (dataset.get(i) > maxVal)  maxVal = dataset.get(i);
			if (dataset.get(i) < minVal)  minVal = dataset.get(i);
		}


		for (int i = 0; i < dataset.size()-1; i++) {

			int curHeight = (int)map(dataset.get(i), minVal, maxVal, 0, H-2);
			int nextHeight = (int)map(dataset.get(i+1), minVal, maxVal, 0, H-2);


			if (dataset.get(i+1) >= dataset.get(i)) {
				g.setColor(Color.blue);
			} else {
				g.setColor(Color.red);
			}

			g.drawLine((int)(i*collumnW)-cameraX,(int)(H-curHeight),(int)((i+1)*collumnW)-cameraX,(int)(H-nextHeight));

		}

	}

	public void tick() {
		if (totalW > W) {

			cameraX += vx;

			if (cameraX+vx < 0) {
				cameraX = 0;
				vx = 0;
			}
			if (cameraX+vx > totalW - W) {
				vx = 0;
				cameraX = totalW - W;
			}
		}
	}

	public void render() {
		BufferStrategy bs = this.getBufferStrategy();
		if (bs == null) {
			this.createBufferStrategy(3);
			return;
		}

		Graphics g = screen.getGraphics();

		g.setColor(Color.black);
		g.fillRect(0,0,W,H);
		// render //

		drawLines(dataSetScore, g);

		// render //
		g = bs.getDrawGraphics();
		Rectangle scr = getBounds();
		g.setColor(Color.black);
		g.fillRect(scr.x,scr.y,scr.width,scr.height);
		g.drawImage(screen,scr.width/2-W/2,scr.height/2-H/2,W,H,null);

		bs.show();
	}

	public static void main(String[] args) {
		new Visualizer();
	}

	public void run() {
		long lastTime = System.nanoTime();
		double amountOfTicks = 30.0;
		double ns = 1000000000 / amountOfTicks;
		double delta = 0;
		int frames = 0;
		double timer = System.currentTimeMillis();
		isRunning = true;

		while (isRunning) {
			long now = System.nanoTime();
			delta += (now-lastTime) / ns;
			lastTime = now;
			if (delta >= 1) {
				tick();
				render();
				frames++;
				delta--;
			}

			if (System.currentTimeMillis() - timer >= 1000) {
				frameRate = frames;
				frames = 0;
				timer += 1000;
			}
		}
	}

	@Override
	public void keyPressed(KeyEvent e) {    }

	@Override
	public void keyReleased(KeyEvent e) {

		if (e.getKeyCode() == KeyEvent.VK_LEFT) {
			vx -= maxVx;
		}
		if (e.getKeyCode() == KeyEvent.VK_RIGHT) {
			vx += maxVx;
		}
		if (e.getKeyCode() == KeyEvent.VK_SPACE) {
			vx = 0;
		}
	}

	@Override
	public void keyTyped(KeyEvent e) {     }

}



