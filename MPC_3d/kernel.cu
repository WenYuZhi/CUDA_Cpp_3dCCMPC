#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <iostream>
#include <math.h>
#include <stdlib.h> 
#include <fstream>
#include <random>
#include "book.h"

using namespace std;

class Mesh
{
public:
	int nx, ny, nz;
	int tnpts;
	float lx, ly, lz;
	float tf;
	float dx, dy, dz;
	float tao;
	Mesh(int, int, int, int, float, float, float, float);
	void print();
};

Mesh::Mesh(int m_nx, int m_ny, int m_nz, int m_tnpts, float m_tf, float m_lx, float m_ly, float m_lz)
{
	nx = m_nx;
	ny = m_ny;
	nz = m_nz;
	tnpts = m_tnpts;
	tf = m_tf;
	lx = m_lx;
	ly = m_ly;
	lz = m_lz;
	dx = m_lx / float(m_nx - 1);
	dy = m_ly / float(m_ny - 1);
	dz = m_lz / float(m_nz - 1);
	tao = m_tf / float(m_tnpts - 1);
}

void Mesh::print()
{
	cout << "nx = " << nx << ", ";
	cout << "ny = " << ny << ", ";
	cout << "nz = " << nz << ", ";
	cout << "lx = " << lx << ", ";
	cout << "ly = " << ly << ", ";
	cout << "lz = " << lz << ", ";
	cout << "dx = " << dx << ", ";
	cout << "dy = " << dy << ", ";
	cout << "dz = " << dz << ", ";
	cout << "tnpts = " << tnpts << ", ";
	cout << "tao = " << tao << ", ";
	cout << "tf = " << tf << ", ";
}

__device__ void Physicial_Parameters(float T, float *pho, float *Ce, float *lamd)
{
	float Ts = 1462.0, Tl = 1518.0, lamds = 30, lamdl = 50, phos = 7000, phol = 7500, ce = 540.0, L = 265600.0, fs = 0.0;
	if (T<Ts)
	{
		fs = 0;
		*pho = phos;
		*lamd = lamds;
		*Ce = ce;
	}

	if (T >= Ts&&T <= Tl)
	{
		fs = (T - Ts) / (Tl - Ts);
		*pho = fs*phos + (1 - fs)*phol;
		*lamd = fs*lamds + (1 - fs)*lamdl;
		*Ce = ce + L / (Tl - Ts);
	}

	if (T>Tl)
	{
		fs = 1;
		*pho = phol;
		*lamd = lamdl;
		*Ce = ce;
	}
}

__device__ float Boundary_Condition(int j, int section, float dy, float *ccml_zone, float *h_init)
{
	float yposition = j * dy, h = 0.0;
	for (int i = 0; i < section; i++)
	{
		if (yposition >= *(ccml_zone + i) && yposition <= *(ccml_zone + i + 1))
			h = *(h_init + i);
	}
	return h;
}

__global__ void pdesolverkernel(float *T_New, float *T_Last, float *ccml, float *h_init, float Vcast, float T_Cast, float dx, float dy, float dz, float tao, int nx, int ny, int nz, int section, bool disout)
{
	int i = threadIdx.x;
	int m = threadIdx.y;
	int j = blockIdx.x;
	int idx = j * nx * nz + m * nx + i;
	int ND = nx * nz;
	int D = nx;

	float pho, Ce, lamd; // physical parameters pho represents desity, Ce is specific heat and lamd is thermal conductivity
	float a, T_Up, T_Down, T_Right, T_Left, T_Forw, T_Back, h, Tw = 30.0;

	if (disout) {
		Physicial_Parameters(T_Last[idx], &pho, &Ce, &lamd);
		a = (lamd) / (pho * Ce);
		h = Boundary_Condition(j, section, dy, ccml, h_init);
		if (j == 0) //1
		{
			T_New[idx] = T_Cast;
		}

		else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m != 0 && m != (nz - 1)) //10
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx - D];
			T_New[idx] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == 0 && m != 0 && m != (nz - 1)) //11
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx + 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //12
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx - 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //13
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx + D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //14
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx - D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == 0 && m == 0)  //15
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx + 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx + D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == 0 && m == (nz - 1))  //16
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx + 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx - D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == (nx - 1) && m == 0)  //17
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx - 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx + D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == (nx - 1) && m == (nz - 1))  //18
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx - 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx - D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //19
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx + D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //20
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx - D] - 2 * dz * h * (T_Last[idx] - Tw) / lamd;
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == 0 && m == 0) //21
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx + 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx + D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == 0)  //22
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx - 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx + D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == 0 && m == (nz - 1)) //23
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx + 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx - D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == (nz - 1)) //24
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx - 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx - D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == 0 && m != 0 && m != (nz - 1))  //25
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx + 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //26
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx - 1] - 2 * dx * h * (T_Last[idx] - Tw) / lamd;
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else  //27
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}
	}

	else
	{
		Physicial_Parameters(T_New[idx], &pho, &Ce, &lamd);
		a = (lamd) / (pho * Ce);
		h = Boundary_Condition(j, section, dy, ccml, h_init);
		if (j == 0) //1
		{
			T_Last[idx] = T_Cast;
		}

		else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m != 0 && m != (nz - 1)) //10
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == 0 && m != 0 && m != (nz - 1)) //11
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx + 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //12
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx - 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //13
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx + D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //14
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx - D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == 0 && m == 0)  //15
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx + 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx + D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == 0 && m == (nz - 1))  //16
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx + 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx - D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == (nx - 1) && m == 0)  //17
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx - 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx + D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == (nx - 1) && m == (nz - 1))  //18
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx - 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx - D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //19
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx + D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //20
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx - D] - 2 * dz * h * (T_Last[idx] - Tw) / lamd;
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == 0 && m == 0) //21
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx + 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx + D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == 0)  //22
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx - 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx + D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == 0 && m == (nz - 1)) //23
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx + 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx - D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == (nz - 1)) //24
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx - 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx - D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == 0 && m != 0 && m != (nz - 1))  //25
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx + 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //26
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx - 1] - 2 * dx * h * (T_New[idx] - Tw) / lamd;
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else  //27
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}
	}
}
class ContinuousCaster
{
    public:
		int section, coolsection, moldsection;
		float *ccml;
	    ContinuousCaster(int, int, int, float*);
	    ~ContinuousCaster();
	    void print();
};

ContinuousCaster::ContinuousCaster(int section, int coolsection, int moldsection, float* ccml)
{
	this->section = section;
	this->coolsection = coolsection;
	this->moldsection = moldsection;
	this->ccml = new float[section + 1];
	for (int i = 0; i < section + 1; i++)
		this->ccml[i] = ccml[i];
}

ContinuousCaster::~ContinuousCaster()
{
	delete[] ccml;
}

void ContinuousCaster::print()
{
	cout << "section = " << section << " ";
	cout << "coolsection = " << coolsection << " ";
	cout << "moldsection = " << moldsection << " " << endl;
	for (int i = 0; i < section; i++)
		cout << ccml[i] << ", ";
}

class Steel
{
    private:
	    float pho;
	    float ce;
	    float lamda;
    public:
	    void physicalpara(float);
		friend class Temperature;
		friend class Temperature1d;
		friend class Temperature2d;
		friend class TemperatureGPU;
};

void Steel::physicalpara(float T)
{
	float Ts = 1462.0, Tl = 1518.0, lamds = 30, lamdl = 50, phos = 7000, phol = 7500, cel = 540.0, L = 265600.0, fs = 0.0;
	if (T < Ts)
	{
		fs = 0;
		pho = phos;
		lamda = lamds;
		ce = cel;
	}

	if (T >= Ts && T <= Tl)
	{
		fs = (T - Ts) / (Tl - Ts);
		pho = fs * phos + (1 - fs) * phol;
		lamda = fs*lamds + (1 - fs) * lamdl;
		ce = cel + L / (Tl - Ts);
	}

	if (T > Tl)
	{
		fs = 1;
		pho = phol;
		lamda = lamdl;
		ce = cel;
	}
}


class Temperature
{
protected:
	float vcast, T_Cast;
	float h;
	float* T_New;
	float* T_Last;
	float* T_Surface;
	bool disout;
	int nx, ny, nz, tnpts;
	float dx, dy, dz, tao, tf, lx, ly, lz;
	ContinuousCaster* mCasterOne;
	Steel* steel;
public:
	int tstep;
	float* meantemperature;
	float* computetemperature;
	Temperature(Mesh &, float, ContinuousCaster &, Steel &);
	Temperature(const Temperature &);
	~Temperature();
	void differencecalculation3d(float*, int);
	void boundarycondition3d(ContinuousCaster &, float*, int);
	void initcondition3d(float);
	void initcondition3d(float*);
	void print3d(int);
	void print3d();
	void computetemperature3d(float *, int);
	void computemeantemperature3d();
	void setvcast(float, float);
	void operator=(const Temperature &);
	friend class Gradientbasedalgorithm;
};

Temperature::Temperature(Mesh & mesh, float m_vcast, ContinuousCaster & m_CasterOne, Steel & m_steel)
{
	mCasterOne = &m_CasterOne;
	steel = &m_steel;
	nx = mesh.nx;
	ny = mesh.ny;
	nz = mesh.nz;
	tnpts = mesh.tnpts;
	tf = mesh.tf;
	lx = mesh.lx;
	ly = mesh.ly;
	lz = mesh.lz;
	dx = mesh.dx;
	dy = mesh.dy;
	dz = mesh.dz;
	tao = mesh.tao;
	T_New = new float[nx * ny * nz];
	T_Last = new float[nx * ny * nz];
	T_Surface = new float[ny];
	meantemperature = new float[mCasterOne->coolsection];
	vcast = m_vcast;
	tstep = 0;
	disout = true;
}

Temperature::Temperature(const Temperature & m_SteelTemperature)
{
	mCasterOne = m_SteelTemperature.mCasterOne;
	steel = m_SteelTemperature.steel;
	nx = m_SteelTemperature.nx;
	ny = m_SteelTemperature.ny;
	nz = m_SteelTemperature.nz;
	tnpts = m_SteelTemperature.tnpts;
	tf = m_SteelTemperature.tf;
	lx = m_SteelTemperature.lx;
	ly = m_SteelTemperature.ly;
	lz = m_SteelTemperature.lz;
	dx = m_SteelTemperature.lx / float(m_SteelTemperature.nx - 1);
	dy = m_SteelTemperature.ly / float(m_SteelTemperature.ny - 1);
	dz = m_SteelTemperature.lz / float(m_SteelTemperature.nz - 1);
	tao = m_SteelTemperature.tf / float(m_SteelTemperature.tnpts - 1);

	T_New = new float[nx * ny * nz];
	T_Last = new float[nx * ny * nz];
	for (int j = 0; j < ny; j++)
		for (int i = 0; i < nx; i++)
			for (int k = 0; k < nz; k++)
			{
				T_Last[nx * nz * j + nz * i + k] = m_SteelTemperature.T_Last[nx * nz * j + nz * i + k];
				T_New[nx * nz * j + nz * i + k] = m_SteelTemperature.T_New[nx * nz * j + nz * i + k];
			}
	T_Surface = new float[ny];
	for (int j = 0; j < ny; j++)
		T_Surface[j] = m_SteelTemperature.T_Surface[j];

	meantemperature = new float[mCasterOne->coolsection];
	for (int i = 0; i < mCasterOne->coolsection; i++)
		meantemperature[i] = m_SteelTemperature.meantemperature[i];

	vcast = m_SteelTemperature.vcast;
	tstep = m_SteelTemperature.tstep;
	disout = m_SteelTemperature.disout;
	h = m_SteelTemperature.h;
}

Temperature::~Temperature()
{
	delete[] T_New;
	delete[] T_Last;
	delete[] T_Surface;
	delete[] meantemperature;
}

void Temperature::setvcast(float vcast, float T_Cast)
{
	this->vcast = vcast;
	this->T_Cast = T_Cast;
}
void Temperature::differencecalculation3d(float *hinit, int m_predictstep = 1)
{
	float a, Tw = 30.0, T_Up, T_Down, T_Right, T_Left, T_Forw, T_Back, T_Middle;
	for (int p = 0; p < m_predictstep; p++)
	{
		if (disout)
		{
			for (int j = 0; j < ny; j++)
			{
				this->boundarycondition3d(*mCasterOne, hinit, j);
				for (int i = 0; i < nx; i++)
					for (int m = 0; m < nz; m++)
					{
						steel->physicalpara(T_Last[nx * nz * j + nz * i + m]);
						a = steel->lamda / (steel->pho * steel->ce);
						if (j == 0 && i != 0 && i != (nx - 1) && m != 0 && m != (nz - 1)) //1
						{
							T_New[nx * nz * j + nz * i + m] = T_Cast;
						}

						else if (j == 0 && i == 0 && m != 0 && m != (nz - 1)) //2
						{
							T_New[nx * nz * j + nz * i + m] = T_Cast;
						}

						else if (j == 0 && i == (nx - 1) && m != 0 && m != (nz - 1))//3
						{
							T_New[nx * nz * j + nz * i + m] = T_Cast;
						}

						else if (j == 0 && i != 0 && i != (nx - 1) && m == 0) //4
						{
							T_New[nx * nz * j + nz * i + m] = T_Cast;
						}

						else if (j == 0 && i != 0 && i != (nx - 1) && m == (nz - 1)) //5
						{
							T_New[nx * nz * j + nz * i + m] = T_Cast;
						}

						else if (j == 0 && i == 0 && m == 0)  //6
						{
							T_New[nx * nz * j + nz * i + m] = T_Cast;
						}

						else if (j == 0 && i == 0 && m == (nz - 1))  //7
						{
							T_New[nx * nz * j + nz * i + m] = T_Cast;
						}

						else if (j == 0 && i == (nx - 1) && m == 0)  //8
						{
							T_New[nx * nz * j + nz * i + m] = T_Cast;
						}

						else if (j == 0 && i == (nx - 1) && m == (nz - 1)) //9
						{
							T_New[nx * nz * j + nz * i + m] = T_Cast;
						}

						else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m != 0 && m != (nz - 1)) //10
						{
							T_Middle = T_Last[nx * nz * j + nz * i + m];
							T_Up = T_Last[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_Last[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Left = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_Last[nx * nz * j + nz * i + m + 1];
							T_Back = T_Last[nx * nz * j + nz * i + m - 1];
							T_Middle = T_Last[nx * nz * j + nz * i + m];
							T_New[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j == (ny - 1) && i == 0 && m != 0 && m != (nz - 1)) //11
						{
							T_Middle = T_Last[nx * nz * j + nz * i + m];
							T_Up = T_Last[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_Last[nx * nz * j + nz * (i + 1) + m];
							T_Right = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Left = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_Last[nx * nz * j + nz * i + m + 1];
							T_Back = T_Last[nx * nz * j + nz * i + m - 1];
							T_New[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j == (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //12
						{
							T_Middle = T_Last[nx * nz * j + nz * i + m];
							T_Up = T_Last[nx * nz * j + nz * (i - 1) + m];
							T_Down = T_Last[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Left = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_Last[nx * nz * j + nz * i + m + 1];
							T_Back = T_Last[nx * nz * j + nz * i + m - 1];
							T_New[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //13
						{
							T_Middle = T_Last[nx * nz * j + nz * i + m];
							T_Up = T_Last[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_Last[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Left = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_Last[nx * nz * j + nz * i + m + 1];
							T_Back = T_Last[nx * nz * j + nz * i + m + 1];
							T_New[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //14
						{
							T_Middle = T_Last[nx * nz * j + nz * i + m];
							T_Up = T_Last[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_Last[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Left = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_Last[nx * nz * j + nz * i + m - 1];
							T_Back = T_Last[nx * nz * j + nz * i + m - 1];
							T_New[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j == (ny - 1) && i == 0 && m == 0)  //15
						{
							T_Middle = T_Last[nx * nz * j + nz * i + m];
							T_Up = T_Last[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_Last[nx * nz * j + nz * (i + 1) + m];
							T_Right = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Left = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_Last[nx * nz * j + nz * i + m + 1];
							T_Back = T_Last[nx * nz * j + nz * i + m + 1];
							T_New[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j == (ny - 1) && i == 0 && m == (nz - 1))  //16
						{
							T_Middle = T_Last[nx * nz * j + nz * i + m];
							T_Up = T_Last[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_Last[nx * nz * j + nz * (i + 1) + m];
							T_Right = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Left = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_Last[nx * nz * j + nz * i + m - 1];
							T_Back = T_Last[nx * nz * j + nz * i + m - 1];
							T_New[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j == (ny - 1) && i == (nx - 1) && m == 0)  //17
						{
							T_Middle = T_Last[nx * nz * j + nz * i + m];
							T_Up = T_Last[nx * nz * j + nz * (i - 1) + m];
							T_Down = T_Last[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Left = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_Last[nx * nz * j + nz * i + m + 1];
							T_Back = T_Last[nx * nz * j + nz * i + m + 1];
							T_New[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j == (ny - 1) && i == (nx - 1) && m == (nz - 1))  //18
						{
							T_Middle = T_Last[nx * nz * j + nz * i + m];
							T_Up = T_Last[nx * nz * j + nz * (i - 1) + m];
							T_Down = T_Last[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Left = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_Last[nx * nz * j + nz * i + m - 1];
							T_Back = T_Last[nx * nz * j + nz * i + m - 1];
							T_New[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //19
						{
							T_Middle = T_Last[nx * nz * j + nz * i + m];
							T_Up = T_Last[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_Last[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_Last[nx * nz * (j + 1) + nz * i + m];
							T_Left = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_Last[nx * nz * j + nz * i + m + 1];
							T_Back = T_Last[nx * nz * j + nz * i + m + 1];
							T_New[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //20
						{
							T_Middle = T_Last[nx * nz * j + nz * i + m];
							T_Up = T_Last[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_Last[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_Last[nx * nz * (j + 1) + nz * i + m];
							T_Left = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_Last[nx * nz * j + nz * i + m - 1] - 2 * dz * h * (T_Middle - Tw) / steel->lamda;
							T_Back = T_Last[nx * nz * j + nz * i + m - 1];
							T_New[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j != 0 && j != (ny - 1) && i == 0 && m == 0) //21
						{
							T_Middle = T_Last[nx * nz * j + nz * i + m];
							T_Up = T_Last[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_Last[nx * nz * j + nz * (i + 1) + m];
							T_Right = T_Last[nx * nz * (j + 1) + nz * i + m];
							T_Left = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_Last[nx * nz * j + nz * i + m + 1];
							T_Back = T_Last[nx * nz * j + nz * i + m + 1];
							T_New[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == 0)  //22
						{
							T_Middle = T_Last[nx * nz * j + nz * i + m];
							T_Up = T_Last[nx * nz * j + nz * (i - 1) + m];
							T_Down = T_Last[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_Last[nx * nz * (j + 1) + nz * i + m];
							T_Left = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_Last[nx * nz * j + nz * i + m + 1];
							T_Back = T_Last[nx * nz * j + nz * i + m + 1];
							T_New[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j != 0 && j != (ny - 1) && i == 0 && m == (nz - 1)) //23
						{
							T_Middle = T_Last[nx * nz * j + nz * i + m];
							T_Up = T_Last[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_Last[nx * nz * j + nz * (i + 1) + m];
							T_Right = T_Last[nx * nz * (j + 1) + nz * i + m];
							T_Left = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_Last[nx * nz * j + nz * i + m - 1];
							T_Back = T_Last[nx * nz * j + nz * i + m - 1];
							T_New[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == (nz - 1)) //24
						{
							T_Middle = T_Last[nx * nz * j + nz * i + m];
							T_Up = T_Last[nx * nz * j + nz * (i - 1) + m];
							T_Down = T_Last[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_Last[nx * nz * (j + 1) + nz * i + m];
							T_Left = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_Last[nx * nz * j + nz * i + m - 1];
							T_Back = T_Last[nx * nz * j + nz * i + m - 1];
							T_New[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j != 0 && j != (ny - 1) && i == 0 && m != 0 && m != (nz - 1))  //25
						{
							T_Middle = T_Last[nx * nz * j + nz * i + m];
							T_Up = T_Last[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_Last[nx * nz * j + nz * (i + 1) + m];
							T_Right = T_Last[nx * nz * (j + 1) + nz * i + m];
							T_Left = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_Last[nx * nz * j + nz * i + m + 1];
							T_Back = T_Last[nx * nz * j + nz * i + m - 1];
							T_New[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //26
						{
							T_Middle = T_Last[nx * nz * j + nz * i + m];
							T_Up = T_Last[nx * nz * j + nz * (i - 1) + m] - 2 * dx * h * (T_Middle - Tw) / steel->lamda;
							T_Down = T_Last[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_Last[nx * nz * (j + 1) + nz * i + m];
							T_Left = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_Last[nx * nz * j + nz * i + m + 1];
							T_Back = T_Last[nx * nz * j + nz * i + m - 1];
							T_New[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else  //27
						{
							T_Middle = T_Last[nx * nz * j + nz * i + m];
							T_Up = T_Last[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_Last[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_Last[nx * nz * (j + 1) + nz * i + m];
							T_Left = T_Last[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_Last[nx * nz * j + nz * i + m + 1];
							T_Back = T_Last[nx * nz * j + nz * i + m - 1];
							T_New[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}
					}
			}
			for (int k = 0; k < ny; k++)
				//T_Surface[k] = 1558.0f;
				T_Surface[k] = T_New[nx * nz * k + nz * int((nx - 1) / 2) + nz - 1];
		}

		else
		{
			for (int j = 0; j < ny; j++)
			{
				this->boundarycondition3d(*mCasterOne, hinit, j);
				for (int i = 0; i < nx; i++)
					for (int m = 0; m < nz; m++)
					{
						steel->physicalpara(T_Last[nx * nz * j + nz * i + m]);
						a = steel->lamda / (steel->pho * steel->ce);
						if (j == 0 && i != 0 && i != (nx - 1) && m != 0 && m != (nz - 1)) //1
						{
							T_Last[nx * nz * j + nz * i + m] = T_Cast;
						}

						else if (j == 0 && i == 0 && m != 0 && m != (nz - 1)) //2
						{
							T_Last[nx * nz * j + nz * i + m] = T_Cast;
						}

						else if (j == 0 && i == (nx - 1) && m != 0 && m != (nz - 1))//3
						{
							T_Last[nx * nz * j + nz * i + m] = T_Cast;
						}

						else if (j == 0 && i != 0 && i != (nx - 1) && m == 0) //4
						{
							T_Last[nx * nz * j + nz * i + m] = T_Cast;
						}

						else if (j == 0 && i != 0 && i != (nx - 1) && m == (nz - 1)) //5
						{
							T_Last[nx * nz * j + nz * i + m] = T_Cast;
						}

						else if (j == 0 && i == 0 && m == 0)  //6
						{
							T_Last[nx * nz * j + nz * i + m] = T_Cast;
						}

						else if (j == 0 && i == 0 && m == (nz - 1))  //7
						{
							T_Last[nx * nz * j + nz * i + m] = T_Cast;
						}

						else if (j == 0 && i == (nx - 1) && m == 0)  //8
						{
							T_Last[nx * nz * j + nz * i + m] = T_Cast;
						}

						else if (j == 0 && i == (nx - 1) && m == (nz - 1)) //9
						{
							T_Last[nx * nz * j + nz * i + m] = T_Cast;
						}

						else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m != 0 && m != (nz - 1)) //10
						{
							T_Middle = T_New[nx * nz * j + nz * i + m];
							T_Up = T_New[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_New[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Left = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_New[nx * nz * j + nz * i + m + 1];
							T_Back = T_New[nx * nz * j + nz * i + m - 1];
							T_Middle = T_New[nx * nz * j + nz * i + m];
							T_Last[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j == (ny - 1) && i == 0 && m != 0 && m != (nz - 1)) //11
						{
							T_Middle = T_New[nx * nz * j + nz * i + m];
							T_Up = T_New[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_New[nx * nz * j + nz * (i + 1) + m];
							T_Right = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Left = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_New[nx * nz * j + nz * i + m + 1];
							T_Back = T_New[nx * nz * j + nz * i + m - 1];
							T_Last[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j == (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //12
						{
							T_Middle = T_New[nx * nz * j + nz * i + m];
							T_Up = T_New[nx * nz * j + nz * (i - 1) + m];
							T_Down = T_New[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Left = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_New[nx * nz * j + nz * i + m + 1];
							T_Back = T_New[nx * nz * j + nz * i + m - 1];
							T_Last[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //13
						{
							T_Middle = T_New[nx * nz * j + nz * i + m];
							T_Up = T_New[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_New[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Left = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_New[nx * nz * j + nz * i + m + 1];
							T_Back = T_New[nx * nz * j + nz * i + m + 1];
							T_Last[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //14
						{
							T_Middle = T_New[nx * nz * j + nz * i + m];
							T_Up = T_New[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_New[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Left = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_New[nx * nz * j + nz * i + m - 1];
							T_Back = T_New[nx * nz * j + nz * i + m - 1];
							T_Last[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j == (ny - 1) && i == 0 && m == 0)  //15
						{
							T_Middle = T_New[nx * nz * j + nz * i + m];
							T_Up = T_New[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_New[nx * nz * j + nz * (i + 1) + m];
							T_Right = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Left = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_New[nx * nz * j + nz * i + m + 1];
							T_Back = T_New[nx * nz * j + nz * i + m + 1];
							T_Last[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j == (ny - 1) && i == 0 && m == (nz - 1))  //16
						{
							T_Middle = T_New[nx * nz * j + nz * i + m];
							T_Up = T_New[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_New[nx * nz * j + nz * (i + 1) + m];
							T_Right = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Left = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_New[nx * nz * j + nz * i + m - 1];
							T_Back = T_New[nx * nz * j + nz * i + m - 1];
							T_Last[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j == (ny - 1) && i == (nx - 1) && m == 0)  //17
						{
							T_Middle = T_New[nx * nz * j + nz * i + m];
							T_Up = T_New[nx * nz * j + nz * (i - 1) + m];
							T_Down = T_New[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Left = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_New[nx * nz * j + nz * i + m + 1];
							T_Back = T_New[nx * nz * j + nz * i + m + 1];
							T_Last[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j == (ny - 1) && i == (nx - 1) && m == (nz - 1))  //18
						{
							T_Middle = T_New[nx * nz * j + nz * i + m];
							T_Up = T_New[nx * nz * j + nz * (i - 1) + m];
							T_Down = T_New[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Left = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_New[nx * nz * j + nz * i + m - 1];
							T_Back = T_New[nx * nz * j + nz * i + m - 1];
							T_Last[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //19
						{
							T_Middle = T_New[nx * nz * j + nz * i + m];
							T_Up = T_New[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_New[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_New[nx * nz * (j + 1) + nz * i + m];
							T_Left = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_New[nx * nz * j + nz * i + m + 1];
							T_Back = T_New[nx * nz * j + nz * i + m + 1];
							T_Last[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //20
						{
							T_Middle = T_New[nx * nz * j + nz * i + m];
							T_Up = T_New[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_New[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_New[nx * nz * (j + 1) + nz * i + m];
							T_Left = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_New[nx * nz * j + nz * i + m - 1] - 2 * dz * h * (T_Middle - Tw) / steel->lamda;
							T_Back = T_New[nx * nz * j + nz * i + m - 1];
							T_Last[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j != 0 && j != (ny - 1) && i == 0 && m == 0) //21
						{
							T_Middle = T_New[nx * nz * j + nz * i + m];
							T_Up = T_New[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_New[nx * nz * j + nz * (i + 1) + m];
							T_Right = T_New[nx * nz * (j + 1) + nz * i + m];
							T_Left = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_New[nx * nz * j + nz * i + m + 1];
							T_Back = T_New[nx * nz * j + nz * i + m + 1];
							T_Last[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == 0)  //22
						{
							T_Middle = T_New[nx * nz * j + nz * i + m];
							T_Up = T_New[nx * nz * j + nz * (i - 1) + m];
							T_Down = T_New[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_New[nx * nz * (j + 1) + nz * i + m];
							T_Left = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_New[nx * nz * j + nz * i + m + 1];
							T_Back = T_New[nx * nz * j + nz * i + m + 1];
							T_Last[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j != 0 && j != (ny - 1) && i == 0 && m == (nz - 1)) //23
						{
							T_Middle = T_New[nx * nz * j + nz * i + m];
							T_Up = T_New[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_New[nx * nz * j + nz * (i + 1) + m];
							T_Right = T_New[nx * nz * (j + 1) + nz * i + m];
							T_Left = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_New[nx * nz * j + nz * i + m - 1];
							T_Back = T_New[nx * nz * j + nz * i + m - 1];
							T_Last[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == (nz - 1)) //24
						{
							T_Middle = T_New[nx * nz * j + nz * i + m];
							T_Up = T_New[nx * nz * j + nz * (i - 1) + m];
							T_Down = T_New[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_New[nx * nz * (j + 1) + nz * i + m];
							T_Left = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_New[nx * nz * j + nz * i + m - 1];
							T_Back = T_New[nx * nz * j + nz * i + m - 1];
							T_Last[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j != 0 && j != (ny - 1) && i == 0 && m != 0 && m != (nz - 1))  //25
						{
							T_Middle = T_New[nx * nz * j + nz * i + m];
							T_Up = T_New[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_New[nx * nz * j + nz * (i + 1) + m];
							T_Right = T_New[nx * nz * (j + 1) + nz * i + m];
							T_Left = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_New[nx * nz * j + nz * i + m + 1];
							T_Back = T_New[nx * nz * j + nz * i + m - 1];
							T_Last[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //26
						{
							T_Middle = T_New[nx * nz * j + nz * i + m];
							T_Up = T_New[nx * nz * j + nz * (i - 1) + m] - 2 * dx * h * (T_Middle - Tw) / steel->lamda;
							T_Down = T_New[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_New[nx * nz * (j + 1) + nz * i + m];
							T_Left = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_New[nx * nz * j + nz * i + m + 1];
							T_Back = T_New[nx * nz * j + nz * i + m - 1];
							T_Last[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}

						else  //27
						{
							T_Middle = T_New[nx * nz * j + nz * i + m];
							T_Up = T_New[nx * nz * j + nz * (i + 1) + m];
							T_Down = T_New[nx * nz * j + nz * (i - 1) + m];
							T_Right = T_New[nx * nz * (j + 1) + nz * i + m];
							T_Left = T_New[nx * nz * (j - 1) + nz * i + m];
							T_Forw = T_New[nx * nz * j + nz * i + m + 1];
							T_Back = T_New[nx * nz * j + nz * i + m - 1];
							T_Last[nx * nz * j + nz * i + m] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*vcast / dy))*T_Middle
								+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
						}
					}
			}
			for (int k = 0; k < ny; k++)
				//T_Surface[k] = 1558.0f;
				T_Surface[k] = T_Last[nx * nz * k + nz * int((nx - 1) / 2) + nz - 1];
		}
		disout = !disout;
		tstep++;
	}
}

void Temperature::boundarycondition3d(ContinuousCaster & CasterOne, float *hinit, int j)
{
	float yposition = dy * j;
	for (int i = 0; i < CasterOne.section; i++)
		if (yposition >= *(CasterOne.ccml + i) && yposition <= *(CasterOne.ccml + i + 1))
			h = *(hinit + i);
}

void Temperature::initcondition3d(float T_Cast)
{
	tstep = 0;
	for (int j = 0; j < ny; j++)
		for (int i = 0; i < nx; i++)
			for (int k = 0; k < nz; k++)
			{
				T_Last[nx * nz * j + nz * i + k] = T_Cast;
				T_New[nx * nz * j + nz * i + k] = T_Cast;
			}
	disout = true;
}

void Temperature::initcondition3d(float *T_Cast)
{
	tstep = 0;
	for (int j = 0; j < ny; j++)
		for (int i = 0; i < nx; i++)
			for (int k = 0; k < nz; k++)
			{
				T_Last[nx * nz * j + nz * i + k] = T_Cast[nx * nz * j + nz * i + k];
				T_New[nx * nz * j + nz * i + k] = T_Cast[nx * nz * j + nz * i + k];
			}
	disout = true;
}


void Temperature::computetemperature3d(float *measuredpoistion, int measurednumb)
{
	computetemperature = new float[measurednumb];
	for (int i = 0; i < measurednumb; i++)
		for (int j = 0; j < ny; j++)
			if (fabs(j * dy - *(measuredpoistion + i)) <= dy)
				computetemperature[i] = T_Surface[j];
}

void Temperature::computemeantemperature3d()
{
	float y;
	int count = 0;
	for (int i = 0; i < mCasterOne->coolsection; i++)
	{
		meantemperature[i] = 0.0;
		for (int j = 0; j < ny; j++)
		{
			y = j * dy;
			if (y > *((mCasterOne->ccml) + i + mCasterOne->moldsection) && y <= *((mCasterOne->ccml) + i + 1 + +mCasterOne->moldsection))
			{
				meantemperature[i] += T_Surface[j];
				count++;
			}
		}
		meantemperature[i] = meantemperature[i] / count;
		count = 0;
	}
}

void Temperature::print3d(int measurednumb)
{
	if (tstep == 0)
	{
		cout << "lx = " << lx << ", " << "nx = " << nx << ", ";
		cout << "ly = " << ly << ", " << "ny = " << ny << ", ";
		cout << "lz = " << lz << ", " << "nz = " << nz << ", ";
		cout << "casting speed = " << vcast << ", " << endl;
		cout << "dx = " << dx << ", ";
		cout << "dy = " << dy << ", ";
		cout << "dz = " << dz << ", ";
		cout << "time step = " << tao << ", " << endl;
	}
	else
	{
		cout << "computetemperature = " << endl;
		for (int i = 0; i < measurednumb; i++)
			cout << computetemperature[i] << ", ";
		cout << endl;
	}
}

void Temperature::print3d()
{
	if (tstep == 0)
	{
		cout << "lx = " << lx << ", " << "nx = " << nx << ", ";
		cout << "ly = " << ly << ", " << "ny = " << ny << ", ";
		cout << "lz = " << lz << ", " << "nz = " << nz << ", ";
		cout << "casting speed = " << vcast << ", " << endl;
		cout << "dx = " << dx << ", ";
		cout << "dy = " << dy << ", ";
		cout << "dz = " << dz << ", ";
		cout << "time step = " << tao << ", " << endl;
	}
	else
	{
		cout << "tstep = " << tstep << endl;
		cout << "meantemperature = " << endl;
		for (int i = 0; i < mCasterOne->coolsection; i++)
			cout << meantemperature[i] << ", ";
		cout << endl;
	}
}

void Temperature::operator=(const Temperature & m_SteelTemperature)
{
	mCasterOne = m_SteelTemperature.mCasterOne;
	steel = m_SteelTemperature.steel;
	nx = m_SteelTemperature.nx;
	ny = m_SteelTemperature.ny;
	nz = m_SteelTemperature.nz;
	tnpts = m_SteelTemperature.tnpts;
	tf = m_SteelTemperature.tf;
	lx = m_SteelTemperature.lx;
	ly = m_SteelTemperature.ly;
	lz = m_SteelTemperature.lz;
	dx = m_SteelTemperature.lx / float(m_SteelTemperature.nx - 1);
	dy = m_SteelTemperature.ly / float(m_SteelTemperature.ny - 1);
	dz = m_SteelTemperature.lz / float(m_SteelTemperature.nz - 1);
	tao = m_SteelTemperature.tf / float(m_SteelTemperature.tnpts - 1);
	vcast = m_SteelTemperature.vcast;
	T_Cast = m_SteelTemperature.T_Cast;
	delete[] T_New;
	delete[] T_Last;
	delete[] T_Surface;
	delete[] meantemperature;

	T_New = new float[nx * ny * nz];
	T_Last = new float[nx * ny * nz];
	for (int j = 0; j < ny; j++)
		for (int i = 0; i < nx; i++)
			for (int k = 0; k < nz; k++)
			{
				T_Last[nx * nz * j + nz * i + k] = m_SteelTemperature.T_Last[nx * nz * j + nz * i + k];
				T_New[nx * nz * j + nz * i + k] = m_SteelTemperature.T_New[nx * nz * j + nz * i + k];
			}
	T_Surface = new float[ny];
	for (int j = 0; j < ny; j++)
		T_Surface[j] = m_SteelTemperature.T_Surface[j];

	meantemperature = new float[mCasterOne->coolsection];
	for (int i = 0; i < mCasterOne->coolsection; i++)
		meantemperature[i] = m_SteelTemperature.meantemperature[i];

	vcast = m_SteelTemperature.vcast;
	tstep = m_SteelTemperature.tstep;
	disout = m_SteelTemperature.disout;
}

class TemperatureGPU:public Temperature
{
    private:
		float* dev_T_New, *dev_T_Last, *dev_ccml, *dev_h_init, *dev_T_Surface;
    public:
	    TemperatureGPU(Mesh & mesh, float, ContinuousCaster &, Steel &);
		void operator=(const TemperatureGPU & m_SteelTemperature);
	    ~TemperatureGPU();
		void initcondition3d(float*);
		void differencecalculation3d(float* ,int);
	    friend class Gradientbasedalgorithm;
};

TemperatureGPU::TemperatureGPU(Mesh & mesh, float m_vcast, ContinuousCaster & m_CasterOne, Steel & m_steel):Temperature(mesh, m_vcast, m_CasterOne, m_steel)
{
	HANDLE_ERROR(cudaSetDevice(0));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_New, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_Last, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_ccml, (m_CasterOne.section + 1) * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_h_init, m_CasterOne.section * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(dev_ccml, m_CasterOne.ccml, (m_CasterOne.section + 1) * sizeof(float), cudaMemcpyHostToDevice));
}

void TemperatureGPU::initcondition3d(float* T_init)
{
	HANDLE_ERROR(cudaMemcpy(dev_T_Last, T_init, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice));
}

void TemperatureGPU::differencecalculation3d(float *hinit, int m_predictstep = 1)
{
	HANDLE_ERROR(cudaMemcpy(dev_h_init, hinit, (mCasterOne->section) * sizeof(float), cudaMemcpyHostToDevice));
	dim3 threadsPerBlock(nx, nz);
	pdesolverkernel << <ny, threadsPerBlock >> >(dev_T_New, dev_T_Last, dev_ccml, dev_h_init, vcast, T_Cast, dx, dy, dz, tao, nx, ny, nz, mCasterOne->section, disout);
	if (disout)
		HANDLE_ERROR(cudaMemcpy(T_New, dev_T_New, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));
	else
		HANDLE_ERROR(cudaMemcpy(T_New, dev_T_Last, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));
	disout = !disout;
	tstep++;
	for (int j = 0; j < ny; j++)
		T_Surface[j] = T_New[nx * nz * j + nz * int((nx - 1) / 2) + nz - 1];
}

void TemperatureGPU::operator=(const TemperatureGPU & m_SteelTemperature)
{
	Temperature::operator= (m_SteelTemperature);
	cudaFree(dev_T_New);
	cudaFree(dev_T_Last);
	cudaFree(dev_ccml);
	cudaFree(dev_h_init);
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_New, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_Last, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_ccml, (mCasterOne->section + 1) * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_h_init, mCasterOne->section * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(dev_ccml, mCasterOne->ccml, (mCasterOne->section + 1) * sizeof(float), cudaMemcpyHostToDevice));
	if(disout)
	    HANDLE_ERROR(cudaMemcpy(dev_T_Last, m_SteelTemperature.T_New, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice));
	else
	    HANDLE_ERROR(cudaMemcpy(dev_T_New, m_SteelTemperature.T_New, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice));
}

TemperatureGPU::~TemperatureGPU()
{
	cudaFree(dev_T_New);
	cudaFree(dev_T_Last);
	cudaFree(dev_ccml);
	cudaFree(dev_h_init);
}

class Temperature2d
{
private:
	float vcast;
	float h;
	float* T_New;
	float* T_Last;
	float* T_Surface;
	bool disout;
	int nx, ny;
	float dx, dy, tao, tf, lx, ly;
	ContinuousCaster* mCasterOne;
	Steel* steel;
public:
	int tstep;
	int tnpts;
	float* computetemperature;
	Temperature2d(int, int, float, float, float, float, ContinuousCaster &, Steel &);
	~Temperature2d();
	void differencecalculation2d(float*);
	void boundarycondition2d(ContinuousCaster*, float*);
	void initcondition2d();
	void computetemperature2d(float *, int);
	void print2d();
};

Temperature2d::Temperature2d(int m_nx, int m_ny, float m_tao, float m_lx, float m_ly, float m_vcast, ContinuousCaster & m_CasterOne, Steel & m_steel)
{
	mCasterOne = &m_CasterOne;
	steel = &m_steel;
	nx = m_nx;
	ny = m_ny;
	tf = m_CasterOne.ccml[(m_CasterOne.section)] / fabs(m_vcast);
	tao = m_tao;
	tnpts = int(tf / tao + 1);
	lx = m_lx;
	ly = m_ly;
	dx = m_lx / float(m_nx - 1);
	dy = m_ly / float(m_ny - 1);
	T_New = new float[nx * ny];
	T_Last = new float[nx * ny];
	T_Surface = new float[tnpts];
	vcast = fabs(m_vcast);
	tstep = 0;
	disout = true;
}

Temperature2d::~Temperature2d()
{
	delete[] T_New;
	delete[] T_Last;
	delete[] T_Surface;
	delete[] computetemperature;
}

void Temperature2d::differencecalculation2d(float* hinit)
{
	float a, Tw = 30.0, T_Up, T_Down, T_Right, T_Left, T_Middle;
	this->boundarycondition2d(mCasterOne, hinit);
	if (disout == 0)
	{
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				steel->physicalpara(T_Last[i * ny + j]);
				a = tao * steel->lamda / (steel->pho * steel->ce);
				if (i == 0 && j != 0 && j != ny - 1)  //1
				{
					T_Up = T_Last[(i + 1)*ny + j];
					T_Down = T_Last[(i + 1)*ny + j] - 2 * dx * h * (T_Last[i*ny + j] - Tw) / steel->lamda;
					T_Right = T_Last[i*ny + j + 1];
					T_Left = T_Last[i*ny + j - 1];
					T_Middle = T_Last[i*ny + j];
					T_New[i * ny + j] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dy*dy) - 2 * a / (dx*dx))*T_Middle + (a / (dy*dy))*T_Right + (a / (dy*dy))*T_Left;
				}
				else if (i == nx - 1 && j != 0 && j != ny - 1)//2
				{
					T_Up = T_Last[(i - 1)*ny + j] - 2 * dx * h * (T_Last[i*ny + j] - Tw) / steel->lamda;
					T_Down = T_Last[(i - 1)*ny + j];
					T_Right = T_Last[i*ny + j + 1];
					T_Left = T_Last[i*ny + j - 1];
					T_Middle = T_Last[i*ny + j];
					T_New[i * ny + j] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dy*dy) - 2 * a / (dx*dx))*T_Middle + (a / (dy*dy))*T_Right + (a / (dy*dy))*T_Left;
				}
				else if (j == 0 && i != 0 && i != nx - 1)//3
				{
					T_Up = T_Last[(i + 1)*ny + j];
					T_Down = T_Last[(i - 1)*ny + j];
					T_Right = T_Last[i*ny + j + 1];
					T_Left = T_Last[i*ny + j + 1];
					T_Middle = T_Last[i*ny + j];
					T_New[i * ny + j] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dy*dy) - 2 * a / (dx*dx))*T_Middle + (a / (dy*dy))*T_Right + (a / (dy*dy))*T_Left;
				}
				else if (j == ny - 1 && i != 0 && i != nx - 1)//4
				{
					T_Up = T_Last[(i + 1)*ny + j];
					T_Down = T_Last[(i - 1)*ny + j];
					T_Right = T_Last[i*ny + j - 1];
					T_Left = T_Last[i*ny + j - 1];
					T_Middle = T_Last[i*ny + j];
					T_New[i * ny + j] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dy*dy) - 2 * a / (dx*dx))*T_Middle + (a / (dy*dy))*T_Right + (a / (dy*dy))*T_Left;
				}
				else if (i == 0 && j == 0)//5
				{
					T_Up = T_Last[(i + 1)*ny + j];
					T_Down = T_Last[(i + 1)*ny + j];
					T_Right = T_Last[i*ny + j + 1];
					T_Left = T_Last[i*ny + j + 1];
					T_Middle = T_Last[i*ny + j];
					T_New[i * ny + j] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dy*dy) - 2 * a / (dx*dx))*T_Middle + (a / (dy*dy))*T_Right + (a / (dy*dy))*T_Left;
				}
				else if (i == 0 && j == ny - 1)//6
				{
					T_Up = T_Last[(i + 1)*ny + j];
					T_Down = T_Last[(i + 1)*ny + j];
					T_Right = T_Last[i*ny + j - 1];
					T_Left = T_Last[i*ny + j - 1];
					T_Middle = T_Last[i*ny + j];
					T_New[i * ny + j] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dy*dy) - 2 * a / (dx*dx))*T_Middle + (a / (dy*dy))*T_Right + (a / (dy*dy))*T_Left;
				}
				else if (i == nx - 1 && j == 0)//7
				{
					T_Up = T_Last[(i - 1)*ny + j];
					T_Down = T_Last[(i - 1)*ny + j];
					T_Right = T_Last[i*ny + j + 1];
					T_Left = T_Last[i*ny + j + 1];
					T_Middle = T_Last[i*ny + j];
					T_New[i * ny + j] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dy*dy) - 2 * a / (dx*dx))*T_Middle + (a / (dy*dy))*T_Right + (a / (dy*dy))*T_Left;
				}
				else if (i == nx - 1 && j == ny - 1)//8
				{
					T_Up = T_Last[(i - 1)*ny + j];
					T_Down = T_Last[(i - 1)*ny + j];
					T_Right = T_Last[i*ny + j - 1];
					T_Left = T_Last[i*ny + j - 1];
					T_Middle = T_Last[i*ny + j];
					T_New[i * ny + j] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dy*dy) - 2 * a / (dx*dx))*T_Middle + (a / (dy*dy))*T_Right + (a / (dy*dy))*T_Left;
				}
				else//9
				{
					T_Up = T_Last[(i + 1)*ny + j];
					T_Down = T_Last[(i - 1)*ny + j];
					T_Right = T_Last[i*ny + j + 1];
					T_Left = T_Last[i*ny + j - 1];
					T_Middle = T_Last[i*ny + j];
					T_New[i * ny + j] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dy*dy) - 2 * a / (dx*dx))*T_Middle + (a / (dy*dy))*T_Right + (a / (dy*dy))*T_Left;
				}
			}
		}
		T_Surface[tstep] = T_New[int((ny - 1) / 2)];
	}
	else
	{
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				steel->physicalpara(T_New[i * ny + j]);
				a = tao * steel->lamda / (steel->pho * steel->ce);
				if (i == 0 && j != 0 && j != ny - 1)  //1
				{
					T_Up = T_New[(i + 1)*ny + j];
					T_Down = T_New[(i + 1)*ny + j] - 2 * dx*h*(T_New[i*ny + j] - Tw) / steel->lamda;
					T_Right = T_New[i*ny + j + 1];
					T_Left = T_New[i*ny + j - 1];
					T_Middle = T_New[i*ny + j];
					T_Last[i * ny + j] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dy*dy) - 2 * a / (dx*dx))*T_Middle + (a / (dy*dy))*T_Right + (a / (dy*dy))*T_Left;
				}
				else if (i == nx - 1 && j != 0 && j != ny - 1)//2
				{
					T_Up = T_New[(i - 1)*ny + j] - 2 * dx*h*(T_New[i*ny + j] - Tw) / steel->lamda;
					T_Down = T_New[(i - 1)*ny + j];
					T_Right = T_New[i*ny + j + 1];
					T_Left = T_New[i*ny + j - 1];
					T_Middle = T_New[i*ny + j];
					T_Last[i * ny + j] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dy*dy) - 2 * a / (dx*dx))*T_Middle + (a / (dy*dy))*T_Right + (a / (dy*dy))*T_Left;
				}
				else if (j == 0 && i != 0 && i != nx - 1)//3
				{
					T_Up = T_New[(i + 1)*ny + j];
					T_Down = T_New[(i - 1)*ny + j];
					T_Right = T_New[i*ny + j + 1];
					T_Left = T_New[i*ny + j + 1] - 2 * dy*h*(T_New[i*ny + j] - Tw) / steel->lamda;
					T_Middle = T_New[i*ny + j];
					T_Last[i * ny + j] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dy*dy) - 2 * a / (dx*dx))*T_Middle + (a / (dy*dy))*T_Right + (a / (dy*dy))*T_Left;
				}
				else if (j == ny - 1 && i != 0 && i != nx - 1)//4
				{
					T_Up = T_New[(i + 1)*ny + j];
					T_Down = T_New[(i - 1)*ny + j];
					T_Right = T_New[i*ny + j - 1] - 2 * dy*h*(T_New[i*ny + j] - Tw) / steel->lamda;
					T_Left = T_New[i*ny + j - 1];
					T_Middle = T_New[i*ny + j];
					T_Last[i * ny + j] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dy*dy) - 2 * a / (dx*dx))*T_Middle + (a / (dy*dy))*T_Right + (a / (dy*dy))*T_Left;
				}
				else if (i == 0 && j == 0)//5
				{
					T_Up = T_New[(i + 1)*ny + j];
					T_Down = T_New[(i + 1)*ny + j];
					T_Right = T_New[i*ny + j + 1];
					T_Left = T_New[i*ny + j + 1];
					T_Middle = T_New[i*ny + j];
					T_Last[i * ny + j] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dy*dy) - 2 * a / (dx*dx))*T_Middle + (a / (dy*dy))*T_Right + (a / (dy*dy))*T_Left;
				}
				else if (i == 0 && j == ny - 1)//6
				{
					T_Up = T_New[(i + 1)*ny + j];
					T_Down = T_New[(i + 1)*ny + j];
					T_Right = T_New[i*ny + j - 1];
					T_Left = T_New[i*ny + j - 1];
					T_Middle = T_New[i*ny + j];
					T_Last[i * ny + j] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dy*dy) - 2 * a / (dx*dx))*T_Middle + (a / (dy*dy))*T_Right + (a / (dy*dy))*T_Left;
				}
				else if (i == nx - 1 && j == 0)//7
				{
					T_Up = T_New[(i - 1)*ny + j];
					T_Down = T_New[(i - 1)*ny + j];
					T_Right = T_New[i*ny + j + 1];
					T_Left = T_New[i*ny + j + 1];
					T_Middle = T_New[i*ny + j];
					T_Last[i * ny + j] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dy*dy) - 2 * a / (dx*dx))*T_Middle + (a / (dy*dy))*T_Right + (a / (dy*dy))*T_Left;
				}
				else if (i == nx - 1 && j == ny - 1)//8
				{
					T_Up = T_New[(i - 1)*ny + j];
					T_Down = T_New[(i - 1)*ny + j];
					T_Right = T_New[i*ny + j - 1];
					T_Left = T_New[i*ny + j - 1];
					T_Middle = T_New[i*ny + j];
					T_Last[i * ny + j] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dy*dy) - 2 * a / (dx*dx))*T_Middle + (a / (dy*dy))*T_Right + (a / (dy*dy))*T_Left;
				}
				else//9
				{
					T_Up = T_New[(i + 1)*ny + j];
					T_Down = T_New[(i - 1)*ny + j];
					T_Right = T_New[i*ny + j + 1];
					T_Left = T_New[i*ny + j - 1];
					T_Middle = T_New[i*ny + j];
					T_Last[i * ny + j] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dy*dy) - 2 * a / (dx*dx))*T_Middle + (a / (dy*dy))*T_Right + (a / (dy*dy))*T_Left;
				}
			}
		}
		T_Surface[tstep] = T_Last[int((ny - 1) / 2)];
	}
	disout = !disout;
	tstep++;

}

void Temperature2d::boundarycondition2d(ContinuousCaster* CasterOne, float *hinit)
{
	float zposition = tstep * tao * vcast;
	//cout << "z = " << zposition << "," ;
	//cout << tstep << ", " << tao << ", " << vcast  << endl;
	for (int i = 0; i < CasterOne->section; i++)
		if (zposition >= *(CasterOne->ccml + i) && zposition <= *(CasterOne->ccml + i + 1))
			h = *(hinit + i);
	//cout << "h = " << h << endl;
}

void Temperature2d::initcondition2d()
{
	float T_Cast = 1558.0f;
	tstep = 0;
	for (int i = 0; i < nx; i++)
		for (int j = 0; j < ny; j++)
		{
			T_Last[ny * i + j] = T_Cast;
			T_New[ny * i + j] = T_Cast;
		}
	disout = 0;
}

void Temperature2d::computetemperature2d(float *measuredpoistion, int measurednumb)
{
	computetemperature = new float[measurednumb];
	for (int i = 0; i < measurednumb; i++)
		for (int j = 0; j < tnpts; j++)
		{
			if ((fabs(j * vcast * tao) - *(measuredpoistion + i)) <= fabs(tao * vcast))
				computetemperature[i] = T_Surface[j];
		}
	/*cout << endl << "computetemperature = " << endl;
	for (int i = 0; i < measurednumb; i++)
	cout << computetemperature[i] << ", ";
	cout << endl;*/
}

void Temperature2d::print2d()
{
	if (tstep == 1)
	{
		cout << "the length of steel billets = " << lx << ", ";
		cout << "the width of steel billets = " << ly << ", ";
		cout << "casting speed = " << vcast << ", " << endl;
		cout << "dx = " << dx << ", ";
		cout << "dy = " << dy << ", ";
		cout << "time step = " << tao << ", " << endl;
	}
	if (tstep % 100 == 0)
	{
		cout << "tstep = " << tstep << endl;
		for (int j = 0; j < ny; j++)
			cout << T_New[j] << ", ";
		cout << endl;
	}
}

class Temperature1d
{
private:
	float vcast;
	float h;
	float* T_New;
	float* T_Last;
	float* T_Surface;
	bool disout;
	int nx;
	float dx, tao, tf, lx;
	ContinuousCaster* mCasterOne;
	Steel* steel;
public:
	int tstep;
	int tnpts;
	float* computetemperature;
	Temperature1d(int, float, float, float, ContinuousCaster &, Steel &);
	~Temperature1d();
	void differencecalculation1d(float*);
	void boundarycondition1d(ContinuousCaster*, float*);
	void initcondition1d();
	void computetemperature1d(float *, int);
	void print1d(int);
	void operator=(const Temperature1d & m_SteelTemperature1d);
	friend class Gradientbasedalgorithm;
};

Temperature1d::Temperature1d(int m_nx, float m_tao, float m_lx, float m_vcast, ContinuousCaster & m_CasterOne, Steel & m_steel)
{
	mCasterOne = &m_CasterOne;
	steel = &m_steel;
	nx = m_nx;
	tao = m_tao;
	tf = m_CasterOne.ccml[(m_CasterOne.section)] / fabs(m_vcast);
	tnpts = int(tf / tao + 1);
	lx = m_lx;
	dx = m_lx / float(m_nx - 1);
	T_New = new float[nx];
	T_Last = new float[nx];
	T_Surface = new float[tnpts];
	vcast = fabs(m_vcast);
	tstep = 0;
	disout = true;
}

Temperature1d::~Temperature1d()
{
	delete[] T_New;
	delete[] T_Last;
	delete[] T_Surface;
	delete[] computetemperature;
}

void Temperature1d::differencecalculation1d(float* hinit)
{
	float a, Tw = 30.0, T_Up, T_Down, T_Middle;
	this->boundarycondition1d(mCasterOne, hinit);
	if (disout == 0)
	{
		for (int i = 0; i < nx; i++)
		{
			steel->physicalpara(T_Last[i]);
			a = tao * steel->lamda / (steel->pho * steel->ce);
			if (i == 0)  //1
			{
				T_Up = T_Last[i + 1];
				T_Down = T_Last[i + 1] - 2 * dx * h * (T_Last[i] - Tw) / steel->lamda;
				T_Middle = T_Last[i];
				T_New[i] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dx*dx))*T_Middle;
			}
			else if (i == nx - 1)//2
			{
				T_Up = T_Last[i - 1] - 2 * dx * h * (T_Last[i] - Tw) / steel->lamda;
				T_Down = T_Last[i - 1];
				T_Middle = T_Last[i];
				T_New[i] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dx*dx))*T_Middle;
			}
			else//9
			{
				T_Up = T_Last[i + 1];
				T_Down = T_Last[i - 1];
				T_Middle = T_Last[i];
				T_New[i] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dx*dx))*T_Middle;
			}
		}
		T_Surface[tstep] = T_New[0];
	}
	else
	{
		for (int i = 0; i < nx; i++)
		{
			steel->physicalpara(T_New[i]);
			a = tao * steel->lamda / (steel->pho * steel->ce);
			if (i == 0)  //1
			{
				T_Up = T_New[i + 1];
				T_Down = T_New[i + 1] - 2 * dx * h * (T_New[i] - Tw) / steel->lamda;
				T_Middle = T_New[i];
				T_Last[i] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dx*dx))*T_Middle;
			}
			else if (i == nx - 1)//2
			{
				T_Up = T_New[i - 1] - 2 * dx * h * (T_New[i] - Tw) / steel->lamda;
				T_Down = T_New[i - 1];
				T_Middle = T_New[i];
				T_Last[i] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dx*dx))*T_Middle;
			}
			else//9
			{
				T_Up = T_New[i + 1];
				T_Down = T_New[i - 1];
				T_Middle = T_New[i];
				T_Last[i] = (a / (dx*dx))*T_Up + (a / (dx*dx))*T_Down + (1 - 2 * a / (dx*dx))*T_Middle;
			}
		}
		T_Surface[tstep] = T_Last[0];
	}
	disout = !disout;
	tstep++;
}

void Temperature1d::boundarycondition1d(ContinuousCaster* CasterOne, float *hinit)
{
	float zposition = tstep * tao * vcast;
	//cout << "z = " << zposition;
	for (int i = 0; i < CasterOne->section; i++)
		if (zposition >= *(CasterOne->ccml + i) && zposition <= *(CasterOne->ccml + i + 1))
			h = *(hinit + i);
	//cout << "h = " << h << endl;
}

void Temperature1d::initcondition1d()
{
	float T_Cast = 1558.0f;
	tstep = 0;
	for (int i = 0; i < nx; i++)
	{
		T_Last[i] = T_Cast;
		T_New[i] = T_Cast;
	}
	disout = 0;
}

void Temperature1d::computetemperature1d(float *measuredpoistion, int measurednumb)
{
	computetemperature = new float[measurednumb];
	for (int i = 0; i < measurednumb; i++)
	{
		computetemperature[i] = 0.0f;
		for (int j = 0; j < tnpts; j++)
			if ((fabs(j * vcast * tao) - *(measuredpoistion + i)) <= fabs(tao * vcast))
				computetemperature[i] = T_Surface[j];
	}

	/*cout << endl << "computetemperature = " << endl;
	for (int i = 0; i < measurednumb; i++)
	cout << computetemperature[i] << ", ";
	cout << endl;*/
}

void Temperature1d::operator=(const Temperature1d & m_SteelTemperature1d)
{
	mCasterOne = m_SteelTemperature1d.mCasterOne;
	steel = m_SteelTemperature1d.steel;
	nx = m_SteelTemperature1d.nx;
	tnpts = m_SteelTemperature1d.tnpts;
	tf = m_SteelTemperature1d.tf;
	lx = m_SteelTemperature1d.lx;
	dx = m_SteelTemperature1d.dx;
	tao = m_SteelTemperature1d.tao;
	delete[] T_New;
	delete[] T_Last;
	delete[] T_Surface;
	delete[] computetemperature;

	T_New = new float[nx];
	T_Last = new float[nx];
	for (int i = 0; i < nx; i++)
	{
		T_Last[i] = m_SteelTemperature1d.T_Last[i];
		T_New[i] = m_SteelTemperature1d.T_New[i];
	}
	T_Surface = new float[tnpts];
	for (int j = 0; j < tnpts; j++)
		T_Surface[j] = m_SteelTemperature1d.T_Surface[j];

	computetemperature = new float[mCasterOne->coolsection];
	for (int i = 0; i < mCasterOne->coolsection; i++)
		computetemperature[i] = m_SteelTemperature1d.computetemperature[i];

	vcast = m_SteelTemperature1d.vcast;
	tstep = m_SteelTemperature1d.tstep;
	disout = m_SteelTemperature1d.disout;
}

void Temperature1d::print1d(int measurednumb)
{
	if (tstep == 1)
	{
		cout << "the length of steel billets = " << lx << ", ";
		cout << "casting speed = " << vcast << ", " << endl;
		cout << "dx = " << dx << ", ";
		cout << "time step = " << tao << ", ";
		cout << "tf = " << tf << endl;
	}
	if (tstep % 100 == 0)
	{
		cout << "tstep = " << tstep << " time = " << tstep * tao << endl;
		for (int i = 0; i < nx; i++)
			cout << T_New[i] << ", ";
		cout << endl;
	}
}

class Gradientbasedalgorithm
{
public:
	int coolsection;
	int maxiter_time;
	int iter_time;
	float** allmeantemperature;
	float* staticmeantemperature;
	float* taimmeantemperature;
	float** Jacobian;
	float* gradient;
	float* dk;
	float* weight;
	float* costvalue;
	float dh;
	float step;
	ContinuousCaster* mCasterOne;
	Gradientbasedalgorithm(ContinuousCaster &, float*, int);
	~Gradientbasedalgorithm();
	virtual void gradientcalculation();
	void init(float**, float*, float);
	void linesearch();
	void updateh(float*);
	void validation(float*, float*, float*);
	void print();
	void outputdata(Temperature1d &);
};

class Conjugategradient :public Gradientbasedalgorithm
{
public:
	float * dk_1;
	float *gradient_1;
	float beta;
	int iter_times;
	Conjugategradient(ContinuousCaster &, float*, int);
	~Conjugategradient();
	virtual void gradientcalculation();
};

Conjugategradient::Conjugategradient(ContinuousCaster & CasterOne, float* m_taimmeantemperature, int m_maxiter_time = 100) :Gradientbasedalgorithm(CasterOne, m_taimmeantemperature, m_maxiter_time)
{
	dk_1 = new float[coolsection];
	gradient_1 = new float[coolsection];
	iter_times = 0;
}

Conjugategradient::~Conjugategradient()
{
	delete[] dk_1;
	delete[] gradient_1;
}

void Conjugategradient::gradientcalculation()
{
	float beta1, beta2;
	iter_times++;
	for (int i = 0; i < coolsection; i++)
		for (int j = 0; j < coolsection; j++)
			Jacobian[i][j] = (allmeantemperature[i][j] - staticmeantemperature[j]) / dh;
	for (int i = 0; i < coolsection; i++)
	{
		gradient[i] = 0.0;
		for (int j = 0; j < coolsection; j++)
			gradient[i] = gradient[i] + (taimmeantemperature[j] - staticmeantemperature[j]) * Jacobian[i][j];
	}

	for (int i = 0; i < coolsection; i++)
	{
		if (iter_times == 1)
		{
			dk[i] = gradient[i];
			dk_1[i] = dk[i];
			gradient_1[i] = gradient[i];
		}
		else
		{
			beta1 = 0;
			beta2 = 0;
			for (int j = 0; j < coolsection; j++)
			{
				beta1 += gradient[j] * gradient[j];
				beta2 += gradient_1[j] * gradient_1[j];
			}
			beta = beta1 / beta2;
			dk[i] = gradient[i] + beta * dk_1[i];
		}
	}
}

Gradientbasedalgorithm::Gradientbasedalgorithm(ContinuousCaster & CasterOne, float* m_taimmeantemperature, int m_maxiter_time = 2000)
{
	mCasterOne = &CasterOne;
	coolsection = mCasterOne->coolsection;
	maxiter_time = m_maxiter_time;
	iter_time = 0;
	costvalue = new float[maxiter_time];
	allmeantemperature = new float*[mCasterOne->coolsection];
	for (int i = 0; i < mCasterOne->coolsection; i++)
		allmeantemperature[i] = new float[coolsection];
	Jacobian = new float*[mCasterOne->coolsection];
	for (int i = 0; i < mCasterOne->coolsection; i++)
		Jacobian[i] = new float[coolsection];
	staticmeantemperature = new float[coolsection];
	gradient = new float[coolsection];
	dk = new float[coolsection];
	weight = new float[coolsection];
	taimmeantemperature = new float[coolsection];
	for (int i = 0; i < mCasterOne->coolsection; i++)
		taimmeantemperature[i] = m_taimmeantemperature[i];
}

Gradientbasedalgorithm::~Gradientbasedalgorithm()
{
	for (int i = 0; i < mCasterOne->coolsection; i++)
		delete[] allmeantemperature[i];
	delete [] allmeantemperature;
	for (int i = 0; i < mCasterOne->coolsection; i++)
		delete[] Jacobian[i];
	delete[] Jacobian;
	delete[] taimmeantemperature;
	delete[] staticmeantemperature;
	delete[] gradient;
	delete[] dk;
	delete[] weight;
	delete[] costvalue;
}

void Gradientbasedalgorithm::gradientcalculation()
{
	for (int i = 0; i < coolsection; i++)
		for (int j = 0; j < coolsection; j++)
			Jacobian[i][j] = (allmeantemperature[i][j] - staticmeantemperature[j]) / dh;
	for (int i = 0; i < coolsection; i++)
	{
		gradient[i] = 0.0;
		for (int j = 0; j < coolsection; j++)
			gradient[i] = gradient[i] + weight[i] * (taimmeantemperature[j] - staticmeantemperature[j]) * Jacobian[i][j];
		dk[i] = gradient[i];
	}
}

void::Gradientbasedalgorithm::linesearch()
{
	float step1 = 0.0, step2 = 0.0, eps = 0.10;
	for (int i = 0; i < coolsection; i++)
		for (int j = 0; j < coolsection; j++)
		{
			step1 += weight[i] * (staticmeantemperature[i] - taimmeantemperature[i]) * Jacobian[i][j];
			step2 += Jacobian[i][j] * dk[j] * Jacobian[i][j] * dk[j];
		}
	step = fabs(step1 / (step2 + eps));
}

void::Gradientbasedalgorithm::updateh(float*hresult)
{
	for (int i = 0; i < coolsection; i++)
	{
		hresult[i + mCasterOne->moldsection] = hresult[i + mCasterOne->moldsection] + step * dk[i];
		//cout  << hresult[i + mCasterOne->moldsection] << " ";
	}
	//cout << endl;
	costvalue[iter_time] = 0.0;
	for (int i = 0; i < coolsection; i++)
		costvalue[iter_time] += weight[i] * (staticmeantemperature[i] - taimmeantemperature[i]) * (staticmeantemperature[i] - taimmeantemperature[i])/ coolsection;
	costvalue[iter_time] = sqrt(costvalue[iter_time]);
	iter_time++;
}

void Gradientbasedalgorithm::init(float**m_allmeantemperature, float *m_staticmeantemperature, float m_dh)
{
	dh = m_dh;
	for (int i = 0; i < coolsection; i++)
		for (int j = 0; j < coolsection; j++)
			allmeantemperature[i][j] = m_allmeantemperature[i][j];
	for (int i = 0; i < coolsection; i++)
		staticmeantemperature[i] = m_staticmeantemperature[i];
	for (int i = 0; i < coolsection; i++)
		weight[i] = 1.0f;
}

void::Gradientbasedalgorithm::validation(float*m_measuredtemperaturetemp, float*hinit, float*htemp)
{
	float msetemperature = 0.0;
	for (int i = 0; i < coolsection; i++)
		msetemperature += (m_measuredtemperaturetemp[i] - staticmeantemperature[i]) * (m_measuredtemperaturetemp[i] - staticmeantemperature[i]);

	float mseh = 0.0;
	for (int i = 0; i < coolsection; i++)
		mseh += (hinit[i + mCasterOne->moldsection] - htemp[i + mCasterOne->moldsection]) * (hinit[i + mCasterOne->moldsection] - htemp[i + mCasterOne->moldsection]);
	cout << "mseh = " << mseh << ", ";
	cout << "msetemperature = " << msetemperature << endl;
}

void Gradientbasedalgorithm::print()
{
	/*cout << endl;
	cout << "Jacobian = " << endl;
	for (int i = 0; i < coolsection; i++)
	{
	for (int j = 0; j < coolsection; j++)
	cout << Jacobian[i][j] << ",";
	cout << endl;
	}*/

	/*cout << "staticmeantemperature = " << endl;
	for (int i = 0; i < coolsection; i++)
		cout << staticmeantemperature[i] << ", ";
	cout << endl;*/

	cout << "staticmeantemperature - taimmeantemperature = " << endl;
	for (int i = 0; i < coolsection; i++)
		cout << fabs(staticmeantemperature[i] - taimmeantemperature[i]) << ", ";
	cout << endl;

	cout << "Gradient = ";
	for (int i = 0; i < coolsection; i++)
		cout << gradient[i] << ", ";
	cout << endl;
	cout << "step = " << step << endl;
	cout << "itertime = " << iter_time << endl;
	cout << "costvalue = " << costvalue[iter_time - 1] << endl;
}

void Gradientbasedalgorithm::outputdata(Temperature1d & m_SteelTemperature)
{
	ofstream outputfile;
	outputfile.open("C:\\costvalue.txt", ios::app);
	for (int i = 0; i < maxiter_time; i++)
		outputfile << costvalue[i] << endl;
	outputfile.close();
	outputfile.open("C:\\surfacetemperature.txt", ios::app);
	for (int i = 0; i < m_SteelTemperature.tnpts; i++)
		outputfile << (m_SteelTemperature.T_Surface[i]) << endl;
	outputfile.close();
	/*if (m_tstep % 10 == 0)
	{
	outputfile.open("C:\\SteelTemperatureData.txt", ios::app);
	outputfile << m_tstep << endl;
	outputfile << step << ", " << costvalue << endl;
	for (int i = 0; i < coolsection; i++)
	outputfile << staticmeantemperature[i] << ", ";
	outputfile << endl;
	for (int i = 0; i < coolsection; i++)
	outputfile << (staticmeantemperature[i] - taimmeantemperature[i]) << ", ";
	outputfile << endl;
	for (int i = 0; i < coolsection; i++)
	outputfile << gradient[i] << ", ";
	outputfile << endl;
	for (int i = 0; i < coolsection; i++)
	outputfile << hinit[i + mCasterOne->moldsection] << ", ";
	outputfile << endl;
	outputfile << endl;
	outputfile.close();
	}*/

}



class Weightalgorithm
{
public:
	float* weight;
	float* kesai;
	int n_element;
	Weightalgorithm::Weightalgorithm(Gradientbasedalgorithm &);
	Weightalgorithm::~Weightalgorithm();
	void weightcalcualtion();
	float meancalculation(float*, const int);
	float stdcalculation(float*, float, const int);
};

Weightalgorithm::Weightalgorithm(Gradientbasedalgorithm & m_algorithm)
{
	weight = new float[m_algorithm.coolsection];
	kesai = new float[m_algorithm.coolsection];
	n_element = m_algorithm.coolsection;
	cout << "kesai = ";
	for (int i = 0; i < m_algorithm.coolsection; i++)
	{
		kesai[i] = fabs(m_algorithm.taimmeantemperature[i] - m_algorithm.staticmeantemperature[i]);
		cout << kesai[i] << ", ";
	}
	cout << endl;
}

Weightalgorithm::~Weightalgorithm()
{
	delete[] weight;
	delete[] kesai;
}

void Weightalgorithm::weightcalcualtion()
{
	float kesaistd, kesaimean;
	float pai = 4 * atan(1);

	kesaimean = this->meancalculation(kesai, n_element);
	kesaistd = this->stdcalculation(kesai, kesaimean, n_element);
	float D = 1.06 * kesaistd * pow(n_element, -2.0);
	cout << "weight = ";
	for (int i = 0; i < n_element; i++)
	{
		weight[i] = 0.0f;
		for (int j = 0; j < n_element; j++)
		{
			if (i != j)
				weight[i] += exp(-0.5 * pow((kesai[i] - kesai[j]) / D, 2.0)) / (sqrt(2.0) * pai * D * n_element);
		}
		cout << weight[i] << ", ";
	}
}

float Weightalgorithm::meancalculation(float* m_vector, const int n_element)
{
	float sum = 0.0f;
	for (int i = 0; i < n_element; i++)
		sum += fabs(m_vector[i]);
	return (sum / n_element);
}

float Weightalgorithm::stdcalculation(float* m_vector, float mean, const int n_element)
{
	float std = 0.0f;
	for (int i = 0; i < n_element; i++)
		std += (m_vector[i] - mean) * (m_vector[i] - mean);
	return (sqrt(std / n_element));
}

class PSOalgorithm
{
public:
	int popsize;
	int measurednumb;
	int coolsection, moldsection, section;
	int iter_time;
	int labelgbest;
	float *fitnessvalue;
	float **poph;
	float **popv;
	float *pbest;
	float **pbesth;
	float *gbest;
	float c1, c2, omga, vmax;
	float gbestvalue;
	PSOalgorithm::PSOalgorithm(int, int, float, float*, ContinuousCaster &);
	void fitnessevaluation(float*, float*, Temperature &, Temperature &);
	void fitnessevaluation(float*, float*, Temperature1d &);
	void findgbest();
	void updatepoph(float*);
	void updatepoph();
	void initpoph(float*, float);
};

PSOalgorithm::PSOalgorithm(int m_measurednumb, int m_popsize, float rangeh, float *hinit, ContinuousCaster & CasterOne)
{
	coolsection = CasterOne.coolsection;
	moldsection = CasterOne.moldsection;
	section = CasterOne.section;
	measurednumb = m_measurednumb;
	popsize = m_popsize;
	iter_time = 0;
	omga = 0.5f;
	c1 = 0.2f;
	c2 = 0.2f;
	vmax = 5.0f;
	fitnessvalue = new float[popsize];
	gbest = new float[section];
	pbest = new float[popsize];
	pbesth = new float*[popsize];
	for (int i = 0; i < popsize; i++)
		pbesth[i] = new float[section];
	popv = new float*[popsize];
	for (int i = 0; i < popsize; i++)
		popv[i] = new float[section];
	poph = new float*[popsize];
	for (int i = 0; i < popsize; i++)
		poph[i] = new float[section];
	for (int i = 0; i < popsize; i++)
	{
		srand((unsigned int)time(NULL) * i);
		for (int j = 0; j < section; j++)
		{
			if (j < moldsection)
				poph[i][j] = hinit[j];
			else
				poph[i][j] = hinit[j] + rangeh * rand() / float(RAND_MAX);
		}
		//for (int j = 0; j < section; j++)
		//  cout << poph[i][j] << ", ";
		//cout << endl;
	}

	for (int i = 0; i < popsize; i++)
		for (int j = 0; j < section; j++)
			popv[i][j] = rand() / float(RAND_MAX);
}

void PSOalgorithm::fitnessevaluation(float* measuredtemperature, float* measuredpoistion, Temperature & Temperature3dmodellast, Temperature & Temperature3dtemp)
{
	for (int i = 0; i < popsize; i++)
	{
		Temperature3dtemp = Temperature3dmodellast;
		Temperature3dtemp.differencecalculation3d(poph[i]);
		Temperature3dtemp.computetemperature3d(measuredpoistion, measurednumb);
		fitnessvalue[i] = 0.0;
		for (int j = 0; j < measurednumb; j++)
			fitnessvalue[i] += (Temperature3dtemp.computetemperature[j] - measuredtemperature[j]) * (Temperature3dtemp.computetemperature[j] - measuredtemperature[j]);
	}
	iter_time++;
}

void PSOalgorithm::fitnessevaluation(float* measuredtemperature, float* measuredpoistion, Temperature1d & m_Temperature1d)
{
	cout << "fitness value = " << endl;
	for (int i = 0; i < popsize; i++)
	{
		m_Temperature1d.initcondition1d();
		while (m_Temperature1d.tstep < m_Temperature1d.tnpts)
			m_Temperature1d.differencecalculation1d(poph[i]);
		m_Temperature1d.computetemperature1d(measuredpoistion, measurednumb);
		fitnessvalue[i] = 0.0;
		for (int j = 0; j < measurednumb; j++)
		{
			fitnessvalue[i] += (m_Temperature1d.computetemperature[j] - measuredtemperature[j]) * (m_Temperature1d.computetemperature[j] - measuredtemperature[j]);
		}
		cout << fitnessvalue[i] << ", ";
	}
}

void PSOalgorithm::findgbest()
{
	gbestvalue = fitnessvalue[0];
	if (iter_time == 1)
	{
		for (int i = 1; i < popsize; i++)
			pbest[i] = fitnessvalue[i];
	}
	for (int i = 0; i < popsize; i++)
	{
		if (fitnessvalue[i] < gbestvalue)
		{
			gbestvalue = fitnessvalue[i];
			labelgbest = i;
		}
		if (fitnessvalue[i] < pbest[i])
		{
			pbest[i] = fitnessvalue[i];
			for (int j = 0; j < section; j++)
				pbesth[i][j] = poph[i][j];
		}
	}
	cout << "gbest = ";
	for (int i = 0; i < section; i++)
	{
		gbest[i] = poph[labelgbest][i];
		cout << gbest[i] << ", ";
	}
	cout << "gbestvalue = " << gbestvalue << endl;
}

void PSOalgorithm::initpoph(float*hinit, float rangeh)
{
	for (int i = 0; i < popsize; i++)
		for (int j = 0; j < section; j++)
		{
			if (j < moldsection)
				poph[i][j] = hinit[i];
			else
				poph[i][j] = hinit[i] + rangeh * rand() / float(RAND_MAX);
		}
}

void PSOalgorithm::updatepoph(float*hresult)
{
	for (int i = 0; i < popsize; i++)
		for (int j = moldsection; j < section; j++)
		{
			popv[i][j] = omga * popv[i][j] + c1* rand() * (gbest[j] - poph[i][j]) + c2 * rand() * (pbesth[i][j] - poph[i][j]);
			if (fabs(popv[i][j]) < vmax)
				poph[i][j] = poph[i][j] + popv[i][j];
			else
				poph[i][j] = poph[i][j] + popv[i][j] * vmax / fabs(popv[i][j]);
		}
	for (int j = moldsection; j < section; j++)
		hresult[j] = pbest[j];
}

void PSOalgorithm::updatepoph()
{
	for (int i = 0; i < popsize; i++)
		for (int j = moldsection; j < section; j++)
		{
			popv[i][j] = omga * popv[i][j] + c1* rand() * (gbest[j] - poph[i][j]);
			if (fabs(popv[i][j]) < vmax)
				poph[i][j] = poph[i][j] + popv[i][j];
			else
				poph[i][j] = poph[i][j] + popv[i][j] * vmax / fabs(popv[i][j]);
		}
}

class Generatemeasuredtemperature
{
public:
	float* hinit;
	float* measuredposition;
	float noisemean, noisestd;
	int measurednumb;
	void simulationtemperature(Temperature1d &, float*, float*);
	Generatemeasuredtemperature(ContinuousCaster &, int, float, float, float *, float *);
	~Generatemeasuredtemperature();

};

Generatemeasuredtemperature::Generatemeasuredtemperature(ContinuousCaster & mCasterOne, int m_measurednumb, float m_noisemean, float m_noisestd, float* m_hinit, float* m_measuredposition)
{
	measurednumb = m_measurednumb;
	noisemean = m_noisemean;
	noisestd = m_noisestd;
	hinit = new float[mCasterOne.section];
	for (int i = 0; i < mCasterOne.section; i++)
		hinit[i] = m_hinit[i];
	measuredposition = new float[m_measurednumb];
	for (int i = 0; i < m_measurednumb; i++)
		measuredposition[i] = m_measuredposition[i];
}

Generatemeasuredtemperature::~Generatemeasuredtemperature()
{
	delete[] hinit;
	delete[] measuredposition;
}

void Generatemeasuredtemperature::simulationtemperature(Temperature1d & m_SteelTemperature1d, float*measuredtemperature, float*measuredtemperaturetemp)
{
	m_SteelTemperature1d.initcondition1d();
	while (m_SteelTemperature1d.tstep < m_SteelTemperature1d.tnpts)
		m_SteelTemperature1d.differencecalculation1d(hinit);
	m_SteelTemperature1d.computetemperature1d(measuredposition, measurednumb);
	default_random_engine gen;
	normal_distribution<float> randn(noisemean, noisestd);
	for (int j = 0; j < measurednumb; j++)
	{
		measuredtemperature[j] = m_SteelTemperature1d.computetemperature[j] + randn(gen);
		measuredtemperaturetemp[j] = m_SteelTemperature1d.computetemperature[j];
		cout << measuredtemperaturetemp[j] << ", ";
	}
}
int main()
{
	const int section = 12, coolsection = 8, moldsection = 4, measurednumb = 8, predictstep = 2;
	float ccml[section + 1] = { 0.0f,0.2f,0.4f,0.6f,0.8f,1.0925f,2.27f,4.29f,5.831f,9.6065f,13.6090f,19.87014f,28.599f };
	float measuredpoistion[measurednumb] = { 0.9463f, 1.6812f, 3.28f, 5.0605f, 7.7188f, 11.6077f, 16.7395f, 24.235f };
	float hinit[section] = { 1380.0f,1170.0f,980.0f,800.0f,1223.16f,735.05f,424.32f,392.83f,328.94f,281.64f,246.16f,160.96f };
	//float taimmeantemperature[coolsection] = { 966.149841f, 925.864746f, 952.322083f, 932.175537f, 914.607117f, 890.494263f, 870.804443f, 890.595825f };
	float taimmeantemperature[coolsection] = { 956.10f, 926.52f, 951.73f, 925.303f, 892.644f, 843.13f, 792.645f, 781.821f };
	float measuredtemperature[measurednumb] = { 937.141f, 930.948f, 960.807f, 932.294f, 916.781f, 892.089f, 872.358f, 899.282f };
	float *htemp = new float[section];
	float *hresult = new float[section];
	float lx = 0.25f, ly = 1.79f, lz = 0.25f, tf = 1400.0f, vcast = -0.02f, rangeh = 50.0f, T_Cast = 1558.0f, dh = 1.0f, tao = 0.25f;
	int nx = 25, ny = 179, nz = 170, tnpts = 1501, sim_tnpts, maxiter_time = 100;
	int popsize = 30;
	float **allmeantemperature;
	float *staticmeantemperature;
	ContinuousCaster CasterOne = ContinuousCaster(section, coolsection, moldsection, ccml);
	Steel steel;
	Gradientbasedalgorithm Gradientmethod = Gradientbasedalgorithm(CasterOne, taimmeantemperature);
	Conjugategradient Conjugategradientmethod = Conjugategradient(CasterOne, taimmeantemperature);
	PSOalgorithm PSO = PSOalgorithm(measurednumb, popsize, rangeh, hinit, CasterOne);

	clock_t t_start = clock();
	nx = 31;
	ny = 3001;
	nz = 31;
	lx = 0.25;
	ly = 28.599f;
	lz = 0.25;
	tnpts = 20001;
	tf = 4000.0f;
	sim_tnpts = 20001;
	float* T_init = new float[nx * ny * nz];
	for (int i = 0; i < nx * ny * nz; i++)
		T_init[i] = T_Cast;
	Mesh MeshOne = Mesh(nx, ny, nz, tnpts, tf, lx, ly, lz);
	TemperatureGPU SteelTemperature3dmodelGPU = TemperatureGPU(MeshOne, vcast, CasterOne, steel);
	TemperatureGPU SteelTemperature3dtempGPU = TemperatureGPU(MeshOne, vcast, CasterOne, steel);

	allmeantemperature = new float*[CasterOne.coolsection];
	for (int i = 0; i < CasterOne.coolsection; i++)
		allmeantemperature[i] = new float[CasterOne.coolsection];
	staticmeantemperature = new float[CasterOne.coolsection];

	SteelTemperature3dmodelGPU.initcondition3d(T_init);
	SteelTemperature3dmodelGPU.setvcast(vcast, T_Cast);
	while (SteelTemperature3dmodelGPU.tstep <= sim_tnpts)
	{
		if (SteelTemperature3dmodelGPU.tstep % 1000 == 0)
		{
			SteelTemperature3dmodelGPU.computemeantemperature3d();
			SteelTemperature3dmodelGPU.print3d();
		}
		SteelTemperature3dmodelGPU.differencecalculation3d(hinit);
	}

	SteelTemperature3dtempGPU = SteelTemperature3dmodelGPU;
	vcast = -0.03f;
	while (SteelTemperature3dmodelGPU.tstep < tnpts && SteelTemperature3dmodelGPU.tstep > sim_tnpts)
	{
		SteelTemperature3dmodelGPU.setvcast(vcast, T_Cast);
		if (SteelTemperature3dmodelGPU.tstep % 10 == 0)
		{
			for (int i = 0; i < CasterOne.coolsection + 1; i++)
			{
				SteelTemperature3dtempGPU = SteelTemperature3dmodelGPU;
				SteelTemperature3dtempGPU.computemeantemperature3d();
				if (i == CasterOne.coolsection)
				{
					for (int j = 0; j < CasterOne.section; j++)
						htemp[j] = hinit[j];
					for (int p = 0; p < predictstep; p++)
						SteelTemperature3dtempGPU.differencecalculation3d(htemp);
					SteelTemperature3dtempGPU.computemeantemperature3d();
					for (int j = 0; j < CasterOne.coolsection; j++)
						staticmeantemperature[j] = SteelTemperature3dtempGPU.meantemperature[j];
				}
				else
				{
					for (int j = 0; j < CasterOne.section; j++)
						htemp[j] = hinit[j];
					htemp[moldsection + i] = htemp[moldsection + i] + dh;
					for (int p = 0; p < predictstep; p++)
						SteelTemperature3dtempGPU.differencecalculation3d(htemp);
					SteelTemperature3dtempGPU.computemeantemperature3d();
					for (int j = 0; j < CasterOne.coolsection; j++)
						allmeantemperature[i][j] = SteelTemperature3dtempGPU.meantemperature[j];
				}
			}
			Gradientmethod.init(allmeantemperature, staticmeantemperature, dh);
			Gradientmethod.gradientcalculation();
			Gradientmethod.linesearch();
			Gradientmethod.updateh(hinit);
			if (SteelTemperature3dmodelGPU.tstep % 100 == 0)
			{
				Gradientmethod.print();
				cout << "tstep = " << SteelTemperature3dmodelGPU.tstep << endl;
			}
		}
		SteelTemperature3dmodelGPU.differencecalculation3d(hinit);
		SteelTemperature3dmodelGPU.computemeantemperature3d();
	}
	clock_t t_end = clock();
	cout << "The running time is " << (t_end - t_start) << " (ms)" << endl;
}


