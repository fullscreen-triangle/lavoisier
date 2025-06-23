//! Python Bridge for Lavoisier Integration
//! 
//! Provides seamless integration between Buhera scripts and Lavoisier's
//! Python-based AI modules.

use crate::ast::*;
use crate::errors::*;
use pyo3::prelude::*;

/// Bridge between Buhera and Lavoisier Python modules
pub struct PythonBridge {
    lavoisier_module: Option<PyObject>,
}

impl PythonBridge {
    /// Create new Python bridge
    pub fn new() -> Self {
        Self {
            lavoisier_module: None,
        }
    }

    /// Initialize connection to Lavoisier
    pub fn connect_to_lavoisier(&mut self, py: Python) -> BuheraResult<()> {
        let lavoisier = py.import("lavoisier")
            .map_err(|e| BuheraError::PythonError(format!("Failed to import lavoisier: {}", e)))?;
        
        self.lavoisier_module = Some(lavoisier.to_object(py));
        Ok(())
    }

    /// Call Lavoisier function with objective context
    pub fn call_lavoisier_function(
        &self,
        py: Python,
        function_path: &str,
        objective: &str,
        args: Vec<PyObject>,
    ) -> BuheraResult<PyObject> {
        let lavoisier = self.lavoisier_module.as_ref()
            .ok_or_else(|| BuheraError::PythonError("Lavoisier not connected".to_string()))?;

        // Add objective context to function call
        let mut full_args = vec![py.None()]; // Placeholder for objective
        full_args.extend(args);

        let result = lavoisier.call_method(py, function_path, PyTuple::new(py, &full_args), None)
            .map_err(|e| BuheraError::PythonError(e.to_string()))?;

        Ok(result)
    }

    /// Execute Buhera script via Python integration
    pub fn execute_script_with_lavoisier(
        &self,
        py: Python,
        script: &BuheraScript,
    ) -> BuheraResult<PyObject> {
        let lavoisier = self.lavoisier_module.as_ref()
            .ok_or_else(|| BuheraError::PythonError("Lavoisier not connected".to_string()))?;

        // Convert script to Python dict
        let script_dict = self.script_to_python_dict(py, script)?;

        // Call Lavoisier's execute_buhera_script function
        let result = lavoisier.call_method(
            py,
            "execute_buhera_script",
            (script_dict,),
            None
        ).map_err(|e| BuheraError::PythonError(e.to_string()))?;

        Ok(result)
    }

    /// Convert BuheraScript to Python dictionary
    fn script_to_python_dict(&self, py: Python, script: &BuheraScript) -> BuheraResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        
        // Add objective
        dict.set_item("objective", &script.objective.target)?;
        
        // Add evidence priorities
        let evidence_list: Vec<String> = script.objective.evidence_priorities
            .iter()
            .map(|e| format!("{:?}", e))
            .collect();
        dict.set_item("evidence_priorities", evidence_list)?;
        
        // Add phases
        let phase_names: Vec<String> = script.phases
            .iter()
            .map(|p| p.name.clone())
            .collect();
        dict.set_item("phases", phase_names)?;

        Ok(dict.to_object(py))
    }
} 