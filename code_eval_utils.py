import subprocess
import traceback
import numpy as np
import itertools
from typing import List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
def execute_code_sample(candidate: str, test_case: str, timeout: int = 15) -> Tuple[bool, str, str, str, bool]:
    """
    Executes candidate + test_case.
    Returns: (passed, error_type, traceback, printed_output, executed)
    """
    full_code = candidate + "\n" + test_case
    try:
        output = subprocess.check_output(
            ["python3", "-c", full_code],
            stderr=subprocess.STDOUT,
            timeout=timeout
        ).decode("utf-8", errors="ignore").strip()
        
        executed = "EXEC_OK:" in output
        output = output.split("EXEC_OK:")[1].strip()
        return True,executed,output,None, None

    except subprocess.TimeoutExpired:
        return False,False,None,"TimeoutError", "Execution timed out"
    except subprocess.CalledProcessError as e:
        output_lines = e.output.decode("utf-8", errors="ignore").strip().splitlines()
        trace = "\n".join(output_lines)
        out = trace.split("Traceback")[0].strip()
        err_type = output_lines[-1].split(":")[0].strip() if output_lines else "UnknownError"
        executed = any("EXEC_OK:" in line for line in output_lines)
        if executed:
            trace = None
            out_line = next((line for line in output_lines if "EXEC_OK:" in line), None)
            out = out_line.split("EXEC_OK:")[1].strip() if out_line and "EXEC_OK:" in out_line else None
        else:
            out = None
        return False, executed, out,err_type, trace
    except Exception as e:
        return False,False, None, type(e).__name__, traceback.format_exc()
def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k using the analytical formula.
    """

    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])
def evaluate_code_set(
    test_case: str,
    candidates: List[str],
    k_values: List[int],
    num_workers: int = 8,
    timeout: int = 15
):
    """
    Evaluates a list of code samples in parallel and returns:
    - pass@k for each k in k_values
    - execution stats for each sample
    """
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(execute_code_sample, code, test_case, timeout) for code in candidates]
        for i, f in enumerate(futures):
            passed, executed, output, err_type, trace = f.result()
            results.append({
                "code_out" : output,
                "passed": passed,
                "executed": executed,
                "error_type": err_type,
                "traceback": trace
            })

    num_correct = sum(r["passed"] for r in results)
    num_executed = sum(r["executed"] for r in results)
    num_samples = len(candidates)

    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(num_samples, [num_correct], k)[0]
        for k in k_values
    }
    pass_at_k["exec@1"] = np.float64(num_executed / num_samples)
    return pass_at_k, results