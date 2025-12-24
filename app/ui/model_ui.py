import gradio as gr
from app.database.database import SessionLocal
from app.services.model_service import ModelService


def create_model_ui():
    """Create Gradio UI for model management"""

    with gr.Tab("Model Management"):
        gr.Markdown("# 🤖 Model Management")
        gr.Markdown("Manage and deploy trained models")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Register Model from Training Job")
                training_job_id = gr.Number(
                    label="Training Job ID", precision=0, value=1
                )
                model_name = gr.Textbox(
                    label="Model Name", placeholder="my_finetuned_model"
                )
                model_description = gr.Textbox(
                    label="Description (Optional)",
                    placeholder="Model description",
                    lines=3,
                )
                register_model_btn = gr.Button("Register Model", variant="primary")
                register_output = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### Registered Models")
                refresh_models_btn = gr.Button("🔄 Refresh")
                filter_active = gr.Dropdown(
                    choices=["all", "active", "inactive"],
                    label="Filter by Status",
                    value="all",
                )
                models_df = gr.Dataframe(
                    headers=[
                        "ID",
                        "Name",
                        "Base Model",
                        "Path",
                        "Active",
                        "Eval Loss",
                        "Created",
                    ],
                    label="Models",
                    interactive=False,
                )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Model Details")
                detail_model_id = gr.Number(label="Model ID", precision=0)
                view_model_btn = gr.Button("View Details")
                model_details = gr.JSON(label="Model Information")

            with gr.Column():
                gr.Markdown("### Update Model")
                update_model_id = gr.Number(label="Model ID", precision=0)
                update_name = gr.Textbox(label="New Name (Optional)")
                update_desc = gr.Textbox(label="New Description (Optional)", lines=2)
                update_active = gr.Checkbox(label="Active", value=True)
                update_model_btn = gr.Button("Update Model")
                update_output = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Load Model for Inference")
                inference_model_id = gr.Number(label="Model ID", precision=0)
                load_model_btn = gr.Button("Load Model", variant="primary")
                load_output = gr.Textbox(label="Status", interactive=False)

            with gr.Column():
                gr.Markdown("### Delete Model")
                delete_model_id = gr.Number(label="Model ID", precision=0)
                delete_files = gr.Checkbox(
                    label="Delete model files from disk", value=False
                )
                delete_model_btn = gr.Button("Delete Model", variant="stop")
                delete_output = gr.Textbox(label="Status", interactive=False)

        # Event handlers
        def register_model_handler(job_id, name, description):
            if not job_id:
                return "Error: Training Job ID is required"

            db = SessionLocal()
            try:
                result = ModelService.create_model_from_training(
                    db=db,
                    training_job_id=int(job_id),
                    name=name or None,
                    description=description or None,
                )
                return f"✅ Model registered: {result['name']} (ID: {result['id']})"
            except Exception as e:
                return f"❌ Error: {str(e)}"
            finally:
                db.close()

        def refresh_models_handler(active_filter):
            db = SessionLocal()
            try:
                is_active = None
                if active_filter == "active":
                    is_active = True
                elif active_filter == "inactive":
                    is_active = False

                models = ModelService.get_models(db, is_active=is_active)
                data = [
                    [
                        model["id"],
                        model["name"],
                        model["base_model"],
                        model["model_path"],
                        "✓" if model["is_active"] else "✗",
                        model["eval_loss"] if model["eval_loss"] else "N/A",
                        model["created_at"],
                    ]
                    for model in models
                ]
                return data
            except Exception as e:
                return []
            finally:
                db.close()

        def view_model_handler(model_id):
            if not model_id:
                return {}

            db = SessionLocal()
            try:
                model = ModelService.get_model(db, int(model_id))
                if not model:
                    return {"error": "Model not found"}
                return model
            except Exception as e:
                return {"error": str(e)}
            finally:
                db.close()

        def update_model_handler(model_id, name, description, is_active):
            if not model_id:
                return "Error: Model ID is required"

            db = SessionLocal()
            try:
                result = ModelService.update_model(
                    db=db,
                    model_id=int(model_id),
                    name=name or None,
                    description=description or None,
                    is_active=is_active,
                )
                if not result:
                    return "❌ Error: Model not found"
                return f"✅ Model updated: {result['name']}"
            except Exception as e:
                return f"❌ Error: {str(e)}"
            finally:
                db.close()

        def load_model_handler(model_id):
            if not model_id:
                return "Error: Model ID is required"

            db = SessionLocal()
            try:
                result = ModelService.load_model_for_inference(db, int(model_id))
                return f"✅ Model loaded: {result['model_path']}"
            except Exception as e:
                return f"❌ Error: {str(e)}"
            finally:
                db.close()

        def delete_model_handler(model_id, delete_files_flag):
            if not model_id:
                return "Error: Model ID is required"

            db = SessionLocal()
            try:
                success = ModelService.delete_model(
                    db, int(model_id), delete_files=delete_files_flag
                )
                if success:
                    return "✅ Model deleted successfully"
                else:
                    return "❌ Error: Model not found"
            except Exception as e:
                return f"❌ Error: {str(e)}"
            finally:
                db.close()

        # Connect events
        register_model_btn.click(
            register_model_handler,
            inputs=[training_job_id, model_name, model_description],
            outputs=register_output,
        )

        refresh_models_btn.click(
            refresh_models_handler, inputs=filter_active, outputs=models_df
        )

        filter_active.change(
            refresh_models_handler, inputs=filter_active, outputs=models_df
        )

        view_model_btn.click(
            view_model_handler, inputs=detail_model_id, outputs=model_details
        )

        update_model_btn.click(
            update_model_handler,
            inputs=[update_model_id, update_name, update_desc, update_active],
            outputs=update_output,
        )

        load_model_btn.click(
            load_model_handler, inputs=inference_model_id, outputs=load_output
        )

        delete_model_btn.click(
            delete_model_handler,
            inputs=[delete_model_id, delete_files],
            outputs=delete_output,
        )

    return None
