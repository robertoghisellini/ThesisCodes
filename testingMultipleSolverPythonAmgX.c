static char help[] = "Solves a sparse random linear system with KSP.\n\n";

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <stdio.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscmat.h>


static PetscErrorCode PrintToFileMMformat(Mat *A, Vec B)
{
  const char* fname= "matrix.mtx";
  FILE *fin, *fp;
  fin = fopen(fname, "wt");
  PetscInt n, nnz=0, edited;
  PetscFunctionBegin;
  PetscCall(VecGetSize(B, &n));
  edited=0;

  if (!fin)
  {
      printf("Error opening file '%s'\n", fname);
      exit(1);
  }
  fprintf(fin,"%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(fin,"%%%%AMGX sorted rhs\n");
  fprintf(fin,"%d %d %d\n", n, n,(PetscInt) round(0.2*n*n));
  const PetscScalar *values;
  PetscScalar *Barray;
  PetscInt ncols, j, i, ch, nline=0;
  const PetscInt *cols;
  for (i=0; i<n; i++)
  {
    PetscCall(MatGetRow(A[0], i, &ncols, &cols, &values));
    for (j=0; j<ncols; j++)
      {
        fprintf(fin,"%d %d %f\n", i+1, cols[j]+1, values[j]);
        nnz+=1;
      }
    PetscCall(MatRestoreRow(A[0], i, &ncols, &cols, &values));
  }
  PetscCall(VecGetArray(B, &Barray));
  for (j=0; j<n; j++)
      {
        fprintf(fin,"%f\n", Barray[j]);
      }
  PetscCall(VecRestoreArray(B, &Barray));
  fclose(fin);
  fp = fopen(fname, "r+");
  while((ch=fgetc(fp))!=EOF)
  {
    if(ch=='\n')
      nline+=1;
    if(nline==2 && edited==0)
    {
      fprintf(fp,"%d %d %d\n", n, n, nnz);
      edited=1;
    }
  }
  fclose(fp);
  PetscFunctionReturn(0);
}


PetscErrorCode generateRandomLS(Mat *A, Vec B, PetscInt n, PetscLogDouble time)
{
  Vec         diag;
  PetscRandom rnd;
  /*
     Create random context and setting a different seed each time the program is called
  */
  PetscFunctionBegin;
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rnd));
  PetscCall(PetscRandomSetInterval(rnd, -1,1));
  PetscCall(PetscRandomSetFromOptions(rnd));
  PetscCall(PetscRandomSetSeed(rnd, time));
  PetscCall(PetscRandomSeed(rnd));
  /*
     Create random matrix
  */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, n, n, n, n, (PetscInt)PetscFloorReal(0.2*n), NULL, 1, NULL, &A[0]));
  PetscCall(MatSetOption(A[0], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetRandom(A[0], rnd));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &diag));
  PetscCall(VecSetSizes(diag, n, n));
  PetscCall(VecSetFromOptions(diag));
  PetscCall(VecSetRandom(diag, rnd));
  PetscCall(MatDiagonalSet(A[0], diag, INSERT_VALUES));

  /*
     Create random RHS vector
  */
  PetscCall(VecSetSizes(B, n, n));
  PetscCall(VecSetFromOptions(B));
  PetscCall(VecSetRandom(B, rnd));
  PetscCall(PrintToFileMMformat(&A[0],B));

  PetscCall(PetscRandomDestroy(&rnd));
  PetscFunctionReturn(0);

}


PetscErrorCode solve(Mat *A ,Vec B ,PC pc, KSP ksp, PetscReal tolerance, PetscInt n, PetscLogDouble time, char *prefix)
{
  Vec         X, C, res;
  PetscReal   resNorm;
  PetscViewer viewer;
  char title[128];

  PetscFunctionBegin;
  PetscCall(VecDuplicate(B, &X));
  PetscCall(PetscStrncpy(title, prefix, sizeof(title)));
  PetscCall(PetscStrcat(title,"_Bvec.txt"));
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, title, &viewer));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_PYTHON));
  PetscCall(VecView(B, viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPSolve(ksp, B, X));

  PetscCall(VecDuplicate(X, &C));
  PetscCall(VecDuplicate(X, &res));
  PetscCall(MatMult(A[0], X, C));
  PetscCall(VecWAXPY(res,-1, C, B));
  PetscCall(VecNorm(res, NORM_2, &resNorm));

  if (0)
  {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n\nSparse random matrix defining the linear system: \n\n"));
    PetscCall(MatView(A[0], PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n\nApproximate solution vector: \n\n"));
    PetscCall(VecView(X, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n\nResidual: \n\n"));
    PetscCall(VecView(res, PETSC_VIEWER_STDOUT_WORLD));
  }

  /*
     View solver info; we could instead use the option -ksp_view to
     print this info to the screen at the conclusion of KSPSolve().
  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n\nInformations on the solver utilized for the solution: \n\n"));
  PetscCall(KSPView(ksp, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n\nNorm of the residual: %e\n\n", resNorm));

  PetscCall(PetscStrncpy(title, prefix, sizeof(title)));
  PetscCall(PetscStrcat(title,"_Amat.txt"));
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, title, &viewer));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_PYTHON));
  PetscCall(MatView(A[0], viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(PetscStrncpy(title, prefix, sizeof(title)));
  PetscCall(PetscStrcat(title,"_Xvec.txt"));
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, title, &viewer));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_PYTHON));
  PetscCall(VecView(X, viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&res));
  PetscCall(VecDestroy(&C));
  PetscFunctionReturn(0);

}

int main(int argc, char **args)
{
  PetscInt    nlocal, n = 20;
  PetscLogDouble time;
  Vec         B;
  Mat         *A;
  PC          pc;
  KSP         ksp;
  PetscScalar *Barray;
  char name[128];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  PetscCall(PetscMalloc1(1, &A));
  PetscCall(PetscTime(&time));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &B));
  PetscCall(generateRandomLS(&A[0], B, n, time));

  PetscCall(VecGetLocalSize(B, &nlocal));
  PetscCall(VecGetArray(B, &Barray));
  PetscCall(VecRestoreArray(B, &Barray));

  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCall(KSPSetOperators(ksp, A[0], A[0]));
  PetscCall(KSPGetPC(ksp, &pc));

  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(PetscStrncpy(name, "PCBJACOBI_KSPGMRES", sizeof(name)));
  PetscCall(PCSetType(pc, PCBJACOBI));
  PetscCall(KSPSetType(ksp, KSPGMRES));
  PetscCall(KSPSetTolerances(ksp, 1e-05, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(solve(&A[0], B, pc, ksp, 1e-05, n, time, name));

  PetscCall(PetscStrncpy(name, "PCNONE_KSPBCGS", sizeof(name)));
  PetscCall(PCSetType(pc, PCNONE));
  PetscCall(KSPSetType(ksp, KSPBCGS));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(solve(&A[0], B, pc, ksp, 1e-05, n, time, name));

  PetscCall(PetscStrncpy(name, "PCHMG_PCGAMG", sizeof(name)));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCHMG));
  PetscCall(PCHMGSetInnerPCType(pc, PCGAMG));
  PetscCall(PCHMGSetReuseInterpolation(pc, PETSC_TRUE));
  PetscCall(PCHMGSetUseSubspaceCoarsening(pc, PETSC_TRUE));
  PetscCall(PCHMGUseMatMAIJ(pc, PETSC_FALSE));
  PetscCall(PCHMGSetCoarseningComponent(pc, 0));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(solve(&A[0], B, pc, ksp, 1e-05, n, time, name));

  PetscCall(PetscStrncpy(name, "PCNONE_KSPGMRES", sizeof(name)));
  PetscCall(PCSetType(pc, PCNONE));
  PetscCall(KSPSetType(ksp, KSPGMRES));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(solve(&A[0], B, pc, ksp, 1e-05, n, time, name));

  PetscCall(PetscStrncpy(name, "PCBJACOBI_KSPBCGS", sizeof(name)));
  PetscCall(PCSetType(pc, PCBJACOBI));
  PetscCall(KSPSetType(ksp, KSPBCGS));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(solve(&A[0], B, pc, ksp, 1e-05, n, time, name));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&B));
  PetscCall(MatDestroy(&A[0]));
  PetscCall(PetscFree(A));
  PetscCall(PetscFinalize());
  return 0;
}
