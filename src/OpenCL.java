import org.lwjgl.opencl.Util;
import org.lwjgl.opencl.CLMem;
import org.lwjgl.opencl.CLCommandQueue;
import org.lwjgl.BufferUtils;
import org.lwjgl.LWJGLException;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CLProgram;
import org.lwjgl.opencl.CLKernel;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.lang.management.ManagementFactory;
import java.lang.management.RuntimeMXBean;
import java.lang.management.ThreadMXBean;
import java.nio.BufferUnderflowException;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Vector;

import org.lwjgl.opencl.CL;
import org.lwjgl.opencl.CLContext;
import org.lwjgl.opencl.CLDevice;
import org.lwjgl.opencl.CLPlatform;

import static org.lwjgl.opencl.CL10.*;

public class OpenCL {
	// The OpenCL kernel
	static final String source = "kernel void prime(global const long *nums, global const long *primes, global const long *numberOfPrimes, global long *arePrimes) { "
			+ "  unsigned int xid = get_global_id(0);" + "  long num = nums[xid]; " + "  long numPrimes = numberOfPrimes[0];" + "  long isPrime=1;" + "  for(long n=0;n<numPrimes;n++){"
			+ "    long prime=primes[n];" + "    if(prime*prime>num){" + "      break;" + "    }else if (num%prime==0){" + "      isPrime=0;" + "      break;" + "    }" + "  }"
			+ "  arePrimes[xid]=isPrime;" + "}";

	static final long maxSize = Integer.MAX_VALUE / 1000l;

	static boolean firstTime=true;
	
	private static long lastTime;

	private static ThreadMXBean mxbean;
	
	public static void main(String[] args){
		Vector<Long> primes = new Vector<Long>(Arrays.asList(2l, 3l, 5l, 7l, 11l));

		// data buffers to store data)
		LongBuffer pris = BufferUtils.createLongBuffer(Integer.MAX_VALUE / 100);
		LongBuffer numberOfPrimes = BufferUtils.createLongBuffer(1);
		LongBuffer arePrimes = BufferUtils.createLongBuffer((int) maxSize);
		LongBuffer nums = BufferUtils.createLongBuffer((int) maxSize);

		try {
			CL.create();
		} catch (LWJGLException e) {
			e.printStackTrace();
			return ;
		}
		CLPlatform platform = CLPlatform.getPlatforms().get(0);
		List<CLDevice> devices = platform.getDevices(CL_DEVICE_TYPE_GPU);
		
		
		int bestIndex=0;
		long bestMemory=-1;
		for(int n=0;n<devices.size();n++){
			CLDevice device = devices.get(n);
			String name = device.getInfoString(CL_DEVICE_NAME);
			long memory = device.getInfoLong(CL_DEVICE_GLOBAL_MEM_SIZE);
			int maxClock = device.getInfoInt(CL_DEVICE_MAX_CLOCK_FREQUENCY);
			System.out.println("Graphics card "+(n+1)+ " " + name + " has "+memory+
					" bytes of RAM and a max clock frequency of "+ maxClock +" MHz.");
			if(memory>bestMemory){
				bestMemory=memory;
				bestIndex=n;
			}
		}
		
		System.out.println("Using graphics card " + (bestIndex +1) + ".");
		CLDevice device=devices.get(bestIndex);
		CLContext context;
		try {
			context = CLContext.create(platform, devices, null, null, null);
		} catch (LWJGLException e) {
			e.printStackTrace();
			return ;
		}

		CLMem numsMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nums, null);
		CLMem primesMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pris, null);
		CLMem numberOfPrimesMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numberOfPrimes, null);
		CLMem arePrimesMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, arePrimes, null);

		PrintWriter writer;
		try {
			writer = new PrintWriter("primes.txt");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return ;
		}
		for (int n = 0; n < primes.size(); n++) {
			writer.println(primes.get(n));
		}

		CLCommandQueue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, null);

		CLProgram program = clCreateProgramWithSource(context, source, null);
		Util.checkCLError(clBuildProgram(program, device, "", null));
		CLKernel kernel = clCreateKernel(program, "prime", null);
		
		long lastNumberToCheck = 10000000;
		
		while(true) {
			numberOfPrimes.rewind();
			numberOfPrimes.put(primes.size());
			numberOfPrimes.rewind();
			pris.rewind();
			for (Long longish : primes) {
				//System.out.println("-> "+longish);
				pris.put(longish);
			}
			pris.rewind();
			
			clEnqueueWriteBuffer(queue, primesMem, 1, 0, pris, null, null);
			clEnqueueWriteBuffer(queue, numberOfPrimesMem, 1, 0, numberOfPrimes, null, null);
			clFinish(queue);
				
			
			
			long first = primes.get(primes.size()-1)+2;
			long last = first * first - 2;
			long latest;
			do {				
				latest = Math.min((maxSize-1)*2+first, last);
				long count = 1 + (latest - first) / 2;
				nums.rewind();
				for (long n = 0; n < count; n++) {
					nums.put(n * 2 + first);
				}
				nums.rewind();

								
				clEnqueueWriteBuffer(queue, numsMem, 1, 0, nums, null, null);
				clFinish(queue);
	

				// Execution our kernel
				PointerBuffer kernel1DGlobalWorkSize = BufferUtils.createPointerBuffer(1);
				kernel1DGlobalWorkSize.put(0, count);

				kernel.setArg(0, numsMem);
				kernel.setArg(1, primesMem);
				kernel.setArg(2, numberOfPrimesMem);
				kernel.setArg(3, arePrimesMem);

				int result=clEnqueueNDRangeKernel(queue, kernel, 1, null, kernel1DGlobalWorkSize, null, null, null);

				if(result!=CL_SUCCESS){
					System.out.println("Failed Calculation With Result: "+result);
					System.exit(0);
				}
				
				// Read the results memory back into our result buffer
				arePrimes.rewind();
				clEnqueueReadBuffer(queue, arePrimesMem, 1, 0, arePrimes, null, null);
				arePrimes.rewind();
				clFinish(queue);
				
				
				for (int n = 0; n < count; n++) {
					long num = nums.get();
					try{
						if (arePrimes.get() == 1) {
							writer.println(num);
							// System.out.println(num);
							primes.add(num);
						}
					}catch(BufferUnderflowException er){
						er.printStackTrace();
						//System.out.println("current:"+n+" filled:"+count+" capacity:"+arePrimes.capacity()+" num:"+num);
						System.exit(0);
					}
				}
				first = latest + 2;
				
				
				
				try {
					Thread.sleep(0);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				
				
				
				System.out.println("Found all primes less than or equal to "+latest+" that's "+primes.size()+" in total.");
				System.out.println("The last one was "+primes.get(primes.size()-1)+".");
			} while ((latest != last) && (latest < lastNumberToCheck));
			if (latest >= lastNumberToCheck) {
				break;
			}
		}
		
		System.out.println("Finished finding all primes less than or equal to " + lastNumberToCheck + ".");

		// Clean up OpenCL resources
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseMemObject(numsMem);
		clReleaseMemObject(primesMem);
		clReleaseMemObject(numberOfPrimesMem);
		clReleaseMemObject(arePrimesMem);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		CL.destroy();
		writer.close();
	}
	
	static long timeSinceLastCall(){
		if(firstTime){
			firstTime=false;
			mxbean = ManagementFactory.getThreadMXBean();
			lastTime=mxbean.getCurrentThreadCpuTime();
			return-1l; 
		}else{
			long currentTime=mxbean.getCurrentThreadCpuTime();
			long diff=currentTime-lastTime;
			lastTime=currentTime;
			return diff;
		}
	}

	static LongBuffer toLongBuffer(Vector<Long> longs) {
		LongBuffer buf = BufferUtils.createLongBuffer(longs.size());
		buf.rewind();
		for (Long longish : longs) {
			// System.out.println("-> "+longish);
			buf.put(longish);
		}
		buf.rewind();
		return buf;
	}

	static LongBuffer toLongBuffer(long[] longs) {
		LongBuffer buf = BufferUtils.createLongBuffer(longs.length).put(longs);
		buf.rewind();
		return buf;
	}

	static void print(LongBuffer buffer, int limit) {
		for (int i = 0; i < buffer.capacity() && i < limit; i++) {
			System.out.print((buffer.get(i) + " "));
		}
		System.out.println("");
	}

	static void print(LongBuffer buffer) {
		print(buffer, 1000);
	}

}